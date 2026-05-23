#include "cerebro/inference_engine.hpp"
#include "cerebro/model_weights.hpp"
#include <cmath>

namespace cerebro {

using namespace weights;

bool InferenceEngine::has_invalid(const float* arr, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(arr[i]) || std::isinf(arr[i])) return true;
    }
    return false;
}

void InferenceEngine::softmax(float* arr, std::size_t n) noexcept {
    // Numerically stable softmax: subtract max before exponentiation.
    float max_val = arr[0];
    for (std::size_t i = 1; i < n; ++i)
        if (arr[i] > max_val) max_val = arr[i];

    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = expf(arr[i] - max_val);
        sum += arr[i];
    }
    for (std::size_t i = 0; i < n; ++i)
        arr[i] /= sum;
}

bool InferenceEngine::infer(const std::array<float, kNumFeatures>& features,
                            InferenceResult& result) const noexcept {
    if (has_invalid(features.data(), features.size())) return false;

    // Layer 1: [kL1Out=32][kL1In=14] + bias, ReLU
    float h1[kL1Out];
    for (std::size_t j = 0; j < kL1Out; ++j) {
        float acc = kB1[j];
        for (std::size_t i = 0; i < kL1In; ++i)
            acc += kW1[j][i] * features[i];
        h1[j] = relu(acc);
    }

    // Layer 2: [kL2Out=16][kL1Out=32] + bias, ReLU
    float h2[kL2Out];
    for (std::size_t j = 0; j < kL2Out; ++j) {
        float acc = kB2[j];
        for (std::size_t i = 0; i < kL1Out; ++i)
            acc += kW2[j][i] * h1[i];
        h2[j] = relu(acc);
    }

    // Layer 3: [kL3Out=8][kL2Out=16] + bias, linear (softmax below)
    float logits[kL3Out];
    for (std::size_t j = 0; j < kL3Out; ++j) {
        float acc = kB3[j];
        for (std::size_t i = 0; i < kL2Out; ++i)
            acc += kW3[j][i] * h2[i];
        logits[j] = acc;
    }

    for (std::size_t i = 0; i < kL3Out; ++i)
        result.logits[i] = logits[i];

    softmax(logits, kL3Out);

    std::size_t best = 0;
    for (std::size_t i = 1; i < kL3Out; ++i)
        if (logits[i] > logits[best]) best = i;

    result.gesture    = static_cast<Gesture>(best);
    result.confidence = logits[best];
    return true;
}

} // namespace cerebro
