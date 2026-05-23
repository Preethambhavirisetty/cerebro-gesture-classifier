#include "cerebro/gesture_classifier.hpp"

namespace cerebro {

bool GestureClassifier::classify(InferenceResult& result) const noexcept {
    FeatureVec features;
    if (!extractor_.extract(buf_, features)) return false;
    return engine_.infer(features, result);
}

const char* gesture_name(Gesture g) noexcept {
    switch (g) {
        case Gesture::IDLE:             return "IDLE";
        case Gesture::SINGLE_TAP:       return "SINGLE_TAP";
        case Gesture::DOUBLE_TAP:       return "DOUBLE_TAP";
        case Gesture::TRIPLE_TAP:       return "TRIPLE_TAP";
        case Gesture::SHORT_HOLD:       return "SHORT_HOLD";
        case Gesture::LONG_HOLD:        return "LONG_HOLD";
        case Gesture::SHALLOW_NAVIGATE: return "SHALLOW_NAVIGATE";
        case Gesture::DEEP_NAVIGATE:    return "DEEP_NAVIGATE";
        default:                        return "UNKNOWN";
    }
}

} // namespace cerebro
