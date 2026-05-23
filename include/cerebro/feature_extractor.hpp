#pragma once
#include "cerebro/event.hpp"
#include "cerebro/circular_buffer.hpp"
#include <array>

namespace cerebro {

using FeatureVec  = std::array<float, kNumFeatures>;
using EventBuffer = CircularBuffer<Event, kBufferCapacity>;

// Feature indices — must stay in sync with python/trainer/feature_engineer.py
namespace feat {
    constexpr int EVENT_COUNT        = 0;
    constexpr int TAP_COUNT          = 1;
    constexpr int HOLD_COUNT         = 2;
    constexpr int NAV_COUNT          = 3;
    constexpr int MEAN_IEI_MS        = 4;   // mean inter-event interval
    constexpr int STD_IEI_MS         = 5;   // population std of IEI
    constexpr int MAX_HOLD_MS        = 6;
    constexpr int MEAN_HOLD_MS       = 7;
    constexpr int TOTAL_NAV_DEPTH    = 8;
    constexpr int MAX_NAV_DEPTH      = 9;
    constexpr int EVENT_RATE_HZ      = 10;
    constexpr int WINDOW_UTILIZATION = 11;
    constexpr int GESTURE_DURATION   = 12;  // span from oldest to newest in-window event
    constexpr int TAP_REGULARITY     = 13;  // 1 - CV of inter-tap intervals
} // namespace feat

// Normalization divisors — must stay in sync with python/trainer/feature_engineer.py
namespace norm {
    constexpr float EVENT_COUNT   = 64.0f;
    constexpr float TAP_COUNT     = 10.0f;
    constexpr float HOLD_COUNT    =  5.0f;
    constexpr float NAV_COUNT     = 20.0f;
    constexpr float IEI_MS        = 2000.0f;
    constexpr float HOLD_MS       = 3000.0f;
    constexpr float NAV_DEPTH     = 100.0f;
    constexpr float RATE_HZ       = 30.0f;
    constexpr float DURATION_MS   = 2000.0f;
    constexpr float TAP_REG       =  1.0f;
} // namespace norm

class FeatureExtractor {
public:
    // Extracts the 14-element feature vector from the most recent kWindowMs of events.
    // Returns false if the buffer is empty; out is zeroed in that case.
    bool extract(const EventBuffer& buf, FeatureVec& out) const noexcept;

private:
    static float clamp01(float v) noexcept {
        return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    }
};

} // namespace cerebro
