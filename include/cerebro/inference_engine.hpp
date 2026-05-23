#pragma once
#include "cerebro/event.hpp"
#include <array>

namespace cerebro {

// MLP layer sizes
constexpr std::size_t kL1In  = kNumFeatures;  // 14
constexpr std::size_t kL1Out = 32;
constexpr std::size_t kL2Out = 16;
constexpr std::size_t kL3Out = kNumClasses;   //  8

struct InferenceResult {
    Gesture gesture;
    float   confidence;               // softmax probability of winning class [0,1]
    float   logits[kNumClasses];      // raw pre-softmax logits
};

// Stateless 3-layer MLP (14→32→16→8).
// Weights are compile-time constants from include/cerebro/model_weights.hpp.
// No heap allocation anywhere in the call graph.
class InferenceEngine {
public:
    // Returns false if features contain NaN/Inf (sensor corruption guard).
    bool infer(const std::array<float, kNumFeatures>& features,
               InferenceResult& result) const noexcept;

private:
    static float relu(float x) noexcept { return x > 0.0f ? x : 0.0f; }
    static void  softmax(float* arr, std::size_t n) noexcept;
    static bool  has_invalid(const float* arr, std::size_t n) noexcept;
};

} // namespace cerebro
