#include "cerebro/feature_extractor.hpp"
#include <cmath>
#include <cstring>

namespace cerebro {

bool FeatureExtractor::extract(const EventBuffer& buf, FeatureVec& out) const noexcept {
    out.fill(0.0f);
    if (buf.empty()) return false;

    const uint32_t newest_ts = buf.newest().timestamp_ms;
    const uint32_t cutoff_ts = (newest_ts >= kWindowMs) ? newest_ts - kWindowMs : 0u;

    // Stack-allocated working arrays — no heap allocation in hot path.
    constexpr std::size_t kMax = kBufferCapacity;
    uint32_t ts_buf[kMax];
    uint32_t tap_ts[kMax];
    std::size_t ts_n   = 0;
    std::size_t tap_n  = 0;

    uint32_t event_count = 0;
    uint32_t tap_count   = 0;
    uint32_t hold_count  = 0;
    uint32_t nav_count   = 0;

    float max_hold_ms  = 0.0f;
    float sum_hold_ms  = 0.0f;
    float total_nav    = 0.0f;
    float max_nav      = 0.0f;

    for (std::size_t i = 0; i < buf.size(); ++i) {
        const Event& e = buf[i];
        if (e.timestamp_ms < cutoff_ts) continue;

        ++event_count;
        if (ts_n < kMax) ts_buf[ts_n++] = e.timestamp_ms;

        switch (e.type) {
            case EventType::TAP:
                ++tap_count;
                if (tap_n < kMax) tap_ts[tap_n++] = e.timestamp_ms;
                break;
            case EventType::HOLD_START:
                ++hold_count;
                break;
            case EventType::HOLD_END: {
                const float dur = e.value > 0 ? static_cast<float>(e.value) : 0.0f;
                sum_hold_ms += dur;
                if (dur > max_hold_ms) max_hold_ms = dur;
                break;
            }
            case EventType::NAVIGATE: {
                ++nav_count;
                const float depth = e.value >= 0
                    ? static_cast<float>(e.value)
                    : static_cast<float>(-e.value);
                total_nav += depth;
                if (depth > max_nav) max_nav = depth;
                break;
            }
            default:
                break;   // DROPOUT: counted in timing arrays, not in gesture stats
        }
    }

    if (ts_n == 0) return false;

    // Inter-event interval: population mean and std
    float mean_iei = 0.0f;
    float std_iei  = 0.0f;
    if (ts_n > 1) {
        float sum = 0.0f;
        for (std::size_t i = 1; i < ts_n; ++i)
            sum += static_cast<float>(ts_buf[i] - ts_buf[i - 1]);
        mean_iei = sum / static_cast<float>(ts_n - 1);

        float var = 0.0f;
        for (std::size_t i = 1; i < ts_n; ++i) {
            const float d = static_cast<float>(ts_buf[i] - ts_buf[i - 1]) - mean_iei;
            var += d * d;
        }
        var /= static_cast<float>(ts_n - 1);
        std_iei = sqrtf(var);
    }

    // Event rate over the in-window span
    const float span_ms = (ts_n > 1)
        ? static_cast<float>(ts_buf[ts_n - 1] - ts_buf[0])
        : 1.0f;
    const float rate_hz = (span_ms > 0.0f)
        ? static_cast<float>(event_count - 1) / (span_ms / 1000.0f)
        : 0.0f;

    // Mean hold duration
    const float mean_hold = (hold_count > 0)
        ? sum_hold_ms / static_cast<float>(hold_count)
        : 0.0f;

    // Tap regularity: 1 - CV of inter-tap intervals (requires ≥ 3 taps)
    float tap_reg = 0.0f;
    if (tap_n > 2) {
        float sum_iti = 0.0f;
        for (std::size_t i = 1; i < tap_n; ++i)
            sum_iti += static_cast<float>(tap_ts[i] - tap_ts[i - 1]);
        const float mean_iti = sum_iti / static_cast<float>(tap_n - 1);

        float var_iti = 0.0f;
        for (std::size_t i = 1; i < tap_n; ++i) {
            const float d = static_cast<float>(tap_ts[i] - tap_ts[i - 1]) - mean_iti;
            var_iti += d * d;
        }
        var_iti /= static_cast<float>(tap_n - 1);
        const float cv = (mean_iti > 0.0f) ? sqrtf(var_iti) / mean_iti : 1.0f;
        tap_reg = 1.0f - (cv > 1.0f ? 1.0f : cv);
    }

    // Gesture duration: time span covered by in-window events
    const float gesture_dur = (ts_n > 1)
        ? static_cast<float>(ts_buf[ts_n - 1] - ts_buf[0])
        : 0.0f;

    // Window utilization: fraction of buffer capacity used by in-window events
    const float win_util = static_cast<float>(event_count) / static_cast<float>(kBufferCapacity);

    // Pack and normalize
    out[feat::EVENT_COUNT]        = clamp01(static_cast<float>(event_count) / norm::EVENT_COUNT);
    out[feat::TAP_COUNT]          = clamp01(static_cast<float>(tap_count)   / norm::TAP_COUNT);
    out[feat::HOLD_COUNT]         = clamp01(static_cast<float>(hold_count)  / norm::HOLD_COUNT);
    out[feat::NAV_COUNT]          = clamp01(static_cast<float>(nav_count)   / norm::NAV_COUNT);
    out[feat::MEAN_IEI_MS]        = clamp01(mean_iei  / norm::IEI_MS);
    out[feat::STD_IEI_MS]         = clamp01(std_iei   / norm::IEI_MS);
    out[feat::MAX_HOLD_MS]        = clamp01(max_hold_ms / norm::HOLD_MS);
    out[feat::MEAN_HOLD_MS]       = clamp01(mean_hold   / norm::HOLD_MS);
    out[feat::TOTAL_NAV_DEPTH]    = clamp01(total_nav   / norm::NAV_DEPTH);
    out[feat::MAX_NAV_DEPTH]      = clamp01(max_nav     / norm::NAV_DEPTH);
    out[feat::EVENT_RATE_HZ]      = clamp01(rate_hz     / norm::RATE_HZ);
    out[feat::WINDOW_UTILIZATION] = clamp01(win_util);
    out[feat::GESTURE_DURATION]   = clamp01(gesture_dur / norm::DURATION_MS);
    out[feat::TAP_REGULARITY]     = clamp01(tap_reg     / norm::TAP_REG);

    return true;
}

} // namespace cerebro
