#pragma once

// Optional, buffered host timeline for MTP target/draft attribution.
//
// Set LLAMA_SERVER_STEP_TRACE=/absolute/path/to/trace.tsv before starting the
// server. Records use CLOCK_MONOTONIC nanoseconds and Linux thread IDs so that
// an external profiler may correlate them after checking clock domains.
// No device events or synchronization are introduced. The buffer is written
// on clean process exit AND when a response finishes. SIGKILL/abort may
// still lose an unfinished request. No model content, token IDs, logits or
// prompts are stored. Full trace output must not be used for throughput
// baselines due to host timestamp and disk-write overhead.
//
// The enabled mode adds CPU timestamp, mutex and vector overhead. Use disabled
// runs for throughput baselines. Keep exactly one local server process while
// profiling to avoid misleading cross-process timestamps.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

namespace server_step_trace {

struct event {
    int64_t    timestamp_ns;
    int64_t    raw_ns;         // Nearby CLOCK_MONOTONIC_RAW sample for clock calibration.
    int64_t    thread_id;
    const char * phase;
    int32_t    slot_id;
    int64_t    verification_steps_before;
    int64_t    generation_tokens_before;
    int32_t    batch_tokens;
    int32_t    draft_tokens;
};

inline int64_t monotonic_ns() {
#if defined(__linux__)
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) == 0) {
        return int64_t(now.tv_sec)*1000000000LL + now.tv_nsec;
    }
#endif
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline int64_t monotonic_raw_ns() {
#if defined(__linux__)
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &now) == 0) {
        return int64_t(now.tv_sec)*1000000000LL + now.tv_nsec;
    }
#endif
    return monotonic_ns();
}

inline int64_t thread_id() {
#if defined(__linux__)
    return static_cast<int64_t>(syscall(SYS_gettid));
#else
    return static_cast<int64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
#endif
}

class recorder {
public:
    static bool enabled() {
        static const bool result = [] {
            const char * value = std::getenv("LLAMA_SERVER_STEP_TRACE");
            return value != nullptr && *value != '\0';
        }();
        return result;
    }

    static void record(const char * phase, const int32_t slot_id,
                       const int64_t steps_before, const int64_t generated_before,
                       const int32_t batch_tokens = 0, const int32_t draft_tokens = 0) {
        if (!enabled()) {
            return;
        }
        const event e = {
            monotonic_ns(), monotonic_raw_ns(), thread_id(), phase, slot_id,
            steps_before, generated_before, batch_tokens, draft_tokens
        };
        auto & inst = get();
        std::lock_guard<std::mutex> lock(inst.mu_);
        inst.events_.push_back(e);
    }

    // Write a durable snapshot on request completion. Writing the full buffer
    // twice (request_complete, then post_decode_end) ensures profiler SIGINT
    // after the HTTP response does not discard marker-defined decode windows.
    // This method never synchronizes a GPU and is disabled for normal runs.
    static void flush_now() {
        if (!enabled()) {
            return;
        }
        get().flush();
    }

private:
    explicit recorder() : filename_(std::getenv("LLAMA_SERVER_STEP_TRACE")) {
        events_.reserve(4096);
    }

    static recorder & get() {
        static recorder * const instance = new recorder();
        static const int registration = std::atexit(&recorder::flush_at_exit);
        (void) registration;
        return *instance;
    }

    static void flush_at_exit() {
        get().flush();
    }

    void flush() {
        std::lock_guard<std::mutex> lock(mu_);
        FILE * out = std::fopen(filename_.c_str(), "w");
        if (!out) {
            std::fprintf(stderr, "LLAMA_SERVER_STEP_TRACE: cannot write %s\n", filename_.c_str());
            return;
        }
        std::fprintf(out, "mono_ns\tos_tid\tphase\tslot_id\tverification_steps_before\tgeneration_tokens_before\tbatch_tokens\tdraft_tokens\traw_ns\n");
        for (const auto & e : events_) {
            std::fprintf(out, "%lld\t%lld\t%s\t%d\t%lld\t%lld\t%d\t%d\t%lld\n",
                         static_cast<long long>(e.timestamp_ns), static_cast<long long>(e.thread_id),
                         e.phase, e.slot_id,
                         static_cast<long long>(e.verification_steps_before),
                         static_cast<long long>(e.generation_tokens_before),
                         e.batch_tokens, e.draft_tokens,
                         static_cast<long long>(e.raw_ns));
        }
        std::fclose(out);
    }

    std::mutex mu_;
    std::vector<event> events_;
    std::string filename_;
};

inline void mark(const char * phase, const int32_t slot_id,
                 const int64_t steps_before, const int64_t generated_before,
                 const int32_t batch_tokens = 0, const int32_t draft_tokens = 0) {
    recorder::record(phase, slot_id, steps_before, generated_before, batch_tokens, draft_tokens);
}

// Available to the server on final response and last post-decode boundary.
// Avoid using this on a per-token path: it performs blocking host file I/O.
inline void flush_now() {
    recorder::flush_now();
}

} // namespace server_step_trace
