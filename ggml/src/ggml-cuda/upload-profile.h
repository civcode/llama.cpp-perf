#pragma once

// Opt-in, host-side ggml CUDA/HIP buffer-upload diagnostics.
// This file intentionally uses only the C++ standard library so the collector
// can be tested without CUDA/HIP hardware. It never synchronizes a GPU.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace ggml_cuda_upload_profile {

struct upload_stats {
    uint64_t calls       = 0;
    uint64_t bytes       = 0;
    uint64_t wall_ns     = 0;
    uint64_t sync_ns     = 0;
    uint64_t max_bytes   = 0;
    uint64_t max_sync_ns = 0;
};

using upload_key = std::tuple<std::string, int, std::string>;

class collector {
public:
    void add(const char * op, int device, const char * tensor_name, uint64_t bytes, uint64_t wall_ns, uint64_t sync_ns) {
        std::lock_guard<std::mutex> lock(mutex_);
        const upload_key key(op, device, tensor_name && *tensor_name ? tensor_name : "<unnamed>");
        upload_stats & item = stats_[key];
        ++item.calls;
        item.bytes += bytes;
        item.wall_ns += wall_ns;
        item.sync_ns += sync_ns;
        item.max_bytes = std::max(item.max_bytes, bytes);
        item.max_sync_ns = std::max(item.max_sync_ns, sync_ns);
    }

    void print(FILE * output = stderr) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::pair<upload_key, upload_stats>> rows(stats_.begin(), stats_.end());
        std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
            return a.second.sync_ns > b.second.sync_ns;
        });

        std::fprintf(output, "GGML_CUDA_UPLOAD_PROFILE\top\tdevice\ttensor\tcalls\tbytes\twall_us\tsync_us\tmax_bytes\tmax_sync_us\n");
        for (const auto & row : rows) {
            const auto & key = row.first;
            const auto & stat = row.second;
            std::fprintf(output, "GGML_CUDA_UPLOAD_PROFILE\t%s\t%d\t%s\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\n",
                    std::get<0>(key).c_str(), std::get<1>(key), std::get<2>(key).c_str(),
                    static_cast<unsigned long long>(stat.calls),
                    static_cast<unsigned long long>(stat.bytes),
                    static_cast<unsigned long long>(stat.wall_ns / 1000),
                    static_cast<unsigned long long>(stat.sync_ns / 1000),
                    static_cast<unsigned long long>(stat.max_bytes),
                    static_cast<unsigned long long>(stat.max_sync_ns / 1000));
        }
        std::fflush(output);
    }

private:
    mutable std::mutex mutex_;
    std::map<upload_key, upload_stats> stats_;
};

inline bool enabled() {
    static const bool is_enabled = [] {
        const char * env = std::getenv("GGML_CUDA_PROFILE_UPLOADS");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return is_enabled;
}

inline collector & instance() {
    // Leaked intentionally: guaranteed alive during the atexit callback,
    // independently of backend/context destruction ordering.
    static collector * state = new collector();
    return *state;
}

inline void record(const char * op, int device, const char * name, uint64_t bytes,
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point sync_start) {
    const auto finished = std::chrono::steady_clock::now();
    static const bool registered = [] {
        std::atexit([] { instance().print(); });
        return true;
    }();
    (void) registered;
    const auto wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(finished - start).count();
    const auto sync_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(finished - sync_start).count();
    instance().add(op, device, name, bytes,
                   static_cast<uint64_t>(std::max<int64_t>(0, wall_ns)),
                   static_cast<uint64_t>(std::max<int64_t>(0, sync_ns)));
}

}  // namespace ggml_cuda_upload_profile
