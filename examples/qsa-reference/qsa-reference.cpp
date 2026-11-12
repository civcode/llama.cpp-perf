// Opt-in research-only teacher-forced reference logits for QSA sparse fidelity.
// Uses the ordinary target decode path, never the speculative sampler. Both
// the label token and the scored target position come from *fixed input text*.
// Keep raw manifest/logit/reference data in a fresh external output directory.
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

static void usage(int, char ** argv) {
    std::fprintf(stderr,
        "Usage: %s -m model.gguf [model/split/context flags]\n"
        "Environment: LLAMA_QSA_REFERENCE_TEXT_FILE=<heldout text> \n"
        "             LLAMA_QSA_REFERENCE_OUT_DIR=<fresh existing dir> \n"
        "             LLAMA_QSA_REFERENCE_ROWS=4..64 in multiples of 4 (default 32)\n",
        argv[0]);
}

static bool parse_count(const char * input, int & result) {
    if (!input || !*input) return false;
    char * end = nullptr;
    const long n = std::strtol(input, &end, 10);
    if (!end || end == input || *end != '\0' || n < 4 || n > 64 || n % 4 != 0) return false;
    result = int(n);
    return true;
}

static uint64_t prefix_hash_add(uint64_t hash, llama_token token) {
    // Deterministic FNV-1a over a token's four canonical low-to-high bytes.
    // This detects accidental input mismatch; it is NOT a cryptographic digest.
    const uint32_t value = (uint32_t) token;
    for (int i = 0; i < 4; ++i) {
        hash ^= (value >> (8*i)) & 255U;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static std::string hash_hex(uint64_t hash) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

int main(int argc, char ** argv) {
    const char * text_path = std::getenv("LLAMA_QSA_REFERENCE_TEXT_FILE");
    const char * out_path = std::getenv("LLAMA_QSA_REFERENCE_OUT_DIR");
    if (!text_path || !*text_path || !out_path || !*out_path) {
        usage(argc, argv);
        return 2;
    }

    int n_rows = 32;
    if (const char * v = std::getenv("LLAMA_QSA_REFERENCE_ROWS")) {
        if (!parse_count(v, n_rows)) {
            std::fprintf(stderr, "LLAMA_QSA_REFERENCE_ROWS must be a multiple of four from 4..64\n");
            return 2;
        }
    }

    std::ifstream text_stream(text_path, std::ios::binary);
    if (!text_stream) {
        std::fprintf(stderr, "Cannot open held-out input file\n");
        return 2;
    }
    const std::string text((std::istreambuf_iterator<char>(text_stream)),
                            std::istreambuf_iterator<char>());
    if (text_stream.bad() || text.empty() || text.size() > 1024 * 1024) {
        std::fprintf(stderr, "Reference text must be 1 byte..1 MiB\n");
        return 2;
    }

    const std::string out(out_path);
    std::error_code dir_error;
    if (!std::filesystem::is_directory(out, dir_error) || dir_error ||
        !std::filesystem::is_empty(out, dir_error) || dir_error) {
        std::fprintf(stderr, "Reference output must be an existing empty directory\n");
        return 2;
    }

    common_params params;
    params.n_predict = 0;  // This runner never generates or samples.
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BATCHED, usage)) {
        return 2;
    }
    if (params.model.path.empty()) {
        std::fprintf(stderr, "Model path is required\n");
        return 2;
    }

    llama_backend_init();
    auto mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "Cannot load target model\n");
        return 3;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0 || n_vocab > 262144) {
        std::fprintf(stderr, "Unexpected target vocabulary size\n");
        llama_model_free(model);
        return 3;
    }
    const int estimate = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    if (estimate >= 0) {
        std::fprintf(stderr, "Unexpected non-negative token-count query\n");
        llama_model_free(model);
        return 3;
    }
    std::vector<llama_token> tokens(-estimate);
    const int written = llama_tokenize(vocab, text.data(), text.size(),
                                      tokens.data(), tokens.size(), true, true);
    if (written <= n_rows || written > 65536) {
        std::fprintf(stderr, "Reference text token count %d is outside bounds for %d rows\n", written, n_rows);
        llama_model_free(model);
        return 3;
    }
    tokens.resize(written);

    auto cparams = common_context_params_to_llama(params);
    // Keep the user's chosen n_ctx / KV precision / batch and GPU split.
    // This is a correctness diagnostic, not a prefill timing benchmark.
    const uint32_t n_batch = cparams.n_batch;
    if (n_batch < 4 || cparams.n_ubatch < 4) {
        std::fprintf(stderr, "Reference runner requires n_batch and n_ubatch >= 4 for Q=4 QSA\n");
        llama_model_free(model);
        return 3;
    }
    if (cparams.n_ctx != 0 && tokens.size() > cparams.n_ctx) {
        std::fprintf(stderr, "Reference requires %zu positions, but ctx has %u\n",
                     tokens.size(), cparams.n_ctx);
        llama_model_free(model);
        return 3;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "Cannot initialize context\n");
        llama_model_free(model);
        return 3;
    }
    if (llama_n_ubatch(ctx) < 4) {
        std::fprintf(stderr, "Actual physical batch size is too small for Q=4 QSA\n");
        llama_free(ctx);
        llama_model_free(model);
        return 3;
    }
    if (tokens.size() > llama_n_ctx(ctx)) {
        std::fprintf(stderr, "Insufficient actual target context\n");
        llama_free(ctx);
        llama_model_free(model);
        return 3;
    }

    std::ofstream manifest(out + "/manifest.jsonl");
    std::ofstream references(out + "/references.jsonl");
    if (!manifest || !references) {
        std::fprintf(stderr, "Create fresh writable output directory before running\n");
        llama_free(ctx);
        llama_model_free(model);
        return 3;
    }
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    if (!batch) {
        std::fprintf(stderr, "Could not create batch\n");
        llama_free(ctx);
        llama_model_free(model);
        return 3;
    }

    const int32_t last_scored_pos = (int32_t) tokens.size() - 2;
    const int32_t first_scored_pos = last_scored_pos - n_rows + 1;
    uint64_t prefix_hash = UINT64_C(14695981039346656037);
    bool failed = false;
    int32_t pos = 0;
    while (pos <= last_scored_pos && !failed) {
        const bool capture = pos >= first_scored_pos;
        // The optimized GQA12 path is for *four query positions at once*.
        // A single-token target loop would not test the experimental kernel.
        const int32_t nt = capture ? 4
            : std::min<int32_t>((int32_t) std::min<uint32_t>(n_batch, 512),
                                first_scored_pos - pos);
        if (capture && (nt != 4 || pos+nt-1 > last_scored_pos)) {
            std::fprintf(stderr, "Incorrect four-query reference group\n");
            failed = true;
            break;
        }
        llama_batch_ext_clear(batch);
        std::vector<std::string> prefix_hashes;
        for (int32_t i = 0; i < nt; ++i) {
            const auto idx = llama_batch_ext_add_token(batch, 0, tokens[pos + i]);
            const llama_pos token_pos = pos + i;
            if (idx < 0 || !llama_batch_ext_set_pos(batch, idx, &token_pos) ||
                !llama_batch_ext_set_output_logits(batch, idx, capture || i + 1 == nt)) {
                std::fprintf(stderr, "Could not set target batch token or logits flag\n");
                failed = true;
                break;
            }
            prefix_hash = prefix_hash_add(prefix_hash, tokens[pos+i]);
            if (capture) prefix_hashes.push_back(hash_hex(prefix_hash));
        }
        if (failed) break;
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
            std::fprintf(stderr, "Target decode failed at token position %d\n", pos);
            failed = true;
            break;
        }
        if (capture) {
            for (int32_t i = 0; i < nt; ++i) {
                // All four output flags are set, so ith logits row corresponds
                // to the ith causal query (not the last query in the batch).
                const float * logits = llama_get_logits_ith(ctx, i);
                if (!logits) {
                    std::fprintf(stderr, "No target logits for query %d at position %d\n", i, pos+i);
                    failed = true;
                    break;
                }
                const int32_t row_pos = pos+i;
                const uint32_t step = (uint32_t) (row_pos - first_scored_pos);
                const llama_token label = tokens[row_pos+1];
                const std::string filename = "logits-" + std::to_string(step) + ".f32";
                std::ofstream file(out + "/" + filename, std::ios::binary);
                file.write(reinterpret_cast<const char *>(logits),
                           (size_t) n_vocab * sizeof(float));
                file.close();
                if (!file) { failed = true; break; }

                // Strict manifest equality includes the prefix token hash,
                // full consumed-token count and query-group position.
                manifest << "{\"kind\":\"event\",\"step\":" << step
                         << ",\"phase\":\"teacher_forced_logits\",\"seq\":0,\"position\":" << row_pos
                         << ",\"target_ids\":[" << tokens[row_pos] << "],\"draft_ids\":[],"
                         << "\"accepted\":[],\"rollback\":false,\"replay\":false,"
                         << "\"prefix_token_hash64\":\"" << prefix_hashes[i] << "\","
                         << "\"prefix_token_count\":" << row_pos+1 << ","
                         << "\"query_group_first_pos\":" << pos << "}\n";
                manifest << "{\"kind\":\"tensor\",\"step\":" << step
                         << ",\"phase\":\"teacher_forced_logits\",\"seq\":0,\"position\":" << row_pos
                         << ",\"layer\":-1,\"name\":\"logits\",\"dtype\":\"f32\","
                         << "\"shape\":[" << n_vocab << "],\"path\":\"" << filename << "\"}\n";
                references << "{\"step\":" << step
                           << ",\"phase\":\"teacher_forced_logits\",\"seq\":0,\"position\":" << row_pos
                           << ",\"label_token\":" << label
                           << ",\"origin\":\"teacher_forced_reference\"}\n";
                if (!manifest || !references) { failed = true; break; }
            }
        }
        pos += nt;
    }

    llama_batch_ext_free(batch);
    manifest.close();
    references.close();
    llama_free(ctx);
    llama_model_free(model);
    if (failed || !manifest || !references) {
        std::fprintf(stderr, "Reference fixture incomplete; discard output directory\n");
        return 4;
    }
    std::fprintf(stderr,
        "QSA_REFERENCE completed teacher-forced rows=%d reference_tokens=%zu "
        "vocab=%d (not an inference performance benchmark)\n",
        n_rows, tokens.size(), n_vocab);
    return 0;
}
