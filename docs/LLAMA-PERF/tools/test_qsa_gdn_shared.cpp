// Bounded opt-in integration fixture: a real shared-prefix recurrent state.
// Link against the locally built libllama/ggml and pass MODEL and an empty DIR.
#include "llama.h"
#include "llama-qsa-debug.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

static void require(bool ok, const char * what) { if (!ok) throw std::runtime_error(what); }

int main(int argc, char ** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s MODEL EMPTY_DIR\n", argv[0]); return 2; }
    try {
        require(!std::ifstream(std::string(argv[2]) + "/manifest.jsonl").good() &&
                !std::ifstream(std::string(argv[2]) + "/serializer-reference.jsonl").good(),
                "fixture requires a fresh output directory");
        const char * backend_dir = std::getenv("QSA_FIXTURE_BACKEND_DIR");
        if (backend_dir) ggml_backend_load_all_from_path(backend_dir);
        else ggml_backend_load_all();
        llama_backend_init();
        ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name("CUDA0"),
            ggml_backend_dev_by_name("ROCm0"), ggml_backend_dev_by_name("ROCm1"), nullptr};
        require(devices[0] && devices[1] && devices[2], "missing mixed GPU devices");
        float split[3] = {4, 7, 7};
        auto mp = llama_model_default_params();
        mp.devices = devices; mp.tensor_split = split; mp.n_gpu_layers = -1;
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER; mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        mp.lazy_mode = LLAMA_LAZY_MODE_ON;
        auto * model = llama_model_load_from_file(argv[1], mp);
        require(model, "model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 512; cp.n_batch = 32; cp.n_ubatch = 32; cp.n_seq_max = 2;
        cp.kv_unified = true; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        auto * ctx = llama_init_from_model(model, cp);
        require(ctx, "context failed");
        auto * mem = llama_get_memory(ctx);
        auto batch = llama_batch_init(16, 0, 1);
        auto decode = [&](int seq, int pos, const int * tokens, int n) {
            batch.n_tokens = n;
            for (int i = 0; i < n; ++i) {
                batch.token[i] = tokens[i]; batch.pos[i] = pos + i;
                batch.n_seq_id[i] = 1; batch.seq_id[i][0] = seq;
                batch.logits[i] = i == n-1;
            }
            require(llama_decode(ctx, batch) == 0, "decode failed");
        };
        auto snapshot = [&](unsigned step, int seq, int pos) {
            std::string path = std::string(argv[2]) + "/manifest.jsonl";
            FILE * f = std::fopen(path.c_str(), "ab"); require(f, "manifest open");
            int n = std::fprintf(f, "{\"kind\":\"event\",\"step\":%u,\"phase\":\"fixture\",\"seq\":%d,\"position\":%d,\"target_ids\":[1],\"draft_ids\":[],\"accepted\":[],\"rollback\":false,\"rollback_mode\":\"none\",\"replay\":false}\n", step, seq, pos);
            require(n > 0 && std::fclose(f) == 0, "manifest write");
            require(llama_qsa_debug_dump_gdn(ctx, seq, argv[2], step, "fixture", pos), "dump failed");
        };
        require(!llama_qsa_debug_dump_gdn(ctx, 1, argv[2], 99, "fixture", 0), "empty owner accepted");
        require(!llama_qsa_debug_dump_gdn(ctx, 2, argv[2], 99, "fixture", 0), "invalid owner accepted");
        const int prefix[] = {1, 2, 3, 4};
        decode(0, 0, prefix, 4);
        snapshot(0, 0, 3);
        llama_memory_seq_cp(mem, 0, 1, -1, -1);
        snapshot(1, 1, 3);
        const int token[] = {5};
        decode(1, 4, token, 1);
        snapshot(2, 1, 4);
        snapshot(3, 0, 3);
        require(llama_memory_seq_rm(mem, 1, -1, -1), "remove seq1 failed");
        decode(0, 4, token, 1);
        snapshot(4, 0, 4);
        require(!llama_qsa_debug_dump_gdn(ctx, 1, argv[2], 99, "fixture", 0), "removed owner accepted");
        auto bytes = [&](int step, const char * role) {
            std::string path = std::string(argv[2]) + "/gdn-" + std::to_string(step) +
                "-fixture-l1-gdn_" + role + ".bin";
            std::ifstream file(path, std::ios::binary);
            require(file.good(), "missing captured row (set layer filter to 1)");
            return std::string(std::istreambuf_iterator<char>(file), {});
        };
        for (const char * role : {"r", "s", "p"}) {
            const auto prefix_row = bytes(0, role);
            require(!prefix_row.empty() && prefix_row == bytes(1, role), "shared prefix differs");
            require(prefix_row == bytes(3, role), "donor state changed after branch");
            require(bytes(2, role) == bytes(4, role), "branch continuation differs from donor continuation");
        }
        require(bytes(0, "s") != bytes(2, "s"), "continuation did not change S state");
        // A second split uses different continuation tokens, proving the
        // independently owned sequences can subsequently diverge.
        llama_memory_seq_cp(mem, 0, 1, -1, -1);
        snapshot(5, 1, 4);
        const int branch_one[] = {6};
        const int branch_zero[] = {7};
        decode(1, 5, branch_one, 1);
        snapshot(6, 1, 5);
        snapshot(7, 0, 4);
        decode(0, 5, branch_zero, 1);
        snapshot(8, 0, 5);
        for (const char * role : {"r", "s", "p"}) {
            require(bytes(4, role) == bytes(5, role), "second shared prefix differs");
            require(bytes(4, role) == bytes(7, role), "second branch changed donor");
        }
        require(bytes(6, "s") != bytes(8, "s"), "different continuations did not diverge");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        std::puts("shared fixture completed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fixture: %s\n", e.what());
        return 1;
    }
}
