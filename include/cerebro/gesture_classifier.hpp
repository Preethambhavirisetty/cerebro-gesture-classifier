#pragma once
#include "cerebro/event.hpp"
#include "cerebro/circular_buffer.hpp"
#include "cerebro/feature_extractor.hpp"
#include "cerebro/inference_engine.hpp"

namespace cerebro {

// Top-level API. Push events; call classify() at any time to get the current gesture.
class GestureClassifier {
public:
    GestureClassifier() noexcept = default;

    void push_event(const Event& e) noexcept { buf_.push(e); }

    // Runs feature extraction + MLP inference on buffered events.
    // Returns false if the buffer is empty or inference encounters invalid input.
    bool classify(InferenceResult& result) const noexcept;

    void reset() noexcept { buf_.clear(); }

    const EventBuffer& buffer() const noexcept { return buf_; }
    std::size_t        event_count() const noexcept { return buf_.size(); }

private:
    EventBuffer      buf_;
    FeatureExtractor extractor_;
    InferenceEngine  engine_;
};

} // namespace cerebro
