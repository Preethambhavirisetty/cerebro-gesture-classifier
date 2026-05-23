#include "test_runner.hpp"
#include "cerebro/gesture_classifier.hpp"

using namespace cerebro;

static void push_tap(GestureClassifier& gc, uint32_t ts) {
    gc.push_event({ts, 0, EventType::TAP, 0});
}

static void push_nav(GestureClassifier& gc, uint32_t ts, int16_t depth) {
    gc.push_event({ts, depth, EventType::NAVIGATE, 0});
}

static void push_hold(GestureClassifier& gc, uint32_t ts_start, int16_t dur_ms) {
    gc.push_event({ts_start,                       0,      EventType::HOLD_START, 0});
    gc.push_event({ts_start + (uint32_t)dur_ms, dur_ms, EventType::HOLD_END,   0});
}

int main() {
    TEST_SECTION("empty classifier returns false");
    {
        GestureClassifier gc;
        InferenceResult   result{};
        CHECK_FALSE(gc.classify(result));
        CHECK_EQ(gc.event_count(), 0u);
    }

    TEST_SECTION("single event → classify succeeds");
    {
        GestureClassifier gc;
        push_tap(gc, 1000);

        InferenceResult result{};
        CHECK_TRUE(gc.classify(result));
        // With zero-weights the confidence should be ~0.125 (uniform softmax)
        CHECK_IN01(result.confidence);
        CHECK_TRUE(static_cast<uint8_t>(result.gesture) < kNumClasses);
    }

    TEST_SECTION("confidence in [0,1] for any event mix");
    {
        GestureClassifier gc;
        push_tap(gc,  1000);
        push_tap(gc,  1200);
        push_tap(gc,  1400);
        push_hold(gc, 1500, 300);
        push_nav(gc,  1900, 25);

        InferenceResult result{};
        CHECK_TRUE(gc.classify(result));
        CHECK_IN01(result.confidence);

        float logit_sum = 0.0f;
        for (float l : result.logits) logit_sum += l;
        // All logits are 0 with placeholder weights
        CHECK_NEAR(logit_sum, 0.0f, 1e-4f);
    }

    TEST_SECTION("reset clears buffer; classify fails afterward");
    {
        GestureClassifier gc;
        push_tap(gc, 1000);
        gc.reset();

        InferenceResult result{};
        CHECK_FALSE(gc.classify(result));
        CHECK_EQ(gc.event_count(), 0u);
    }

    TEST_SECTION("push_event accumulates into buffer");
    {
        GestureClassifier gc;
        for (uint32_t i = 0; i < 10; ++i)
            push_tap(gc, 1000 + i * 100);

        CHECK_EQ(gc.event_count(), 10u);
        InferenceResult result{};
        CHECK_TRUE(gc.classify(result));
    }

    TEST_SECTION("buffer cap at kBufferCapacity; oldest evicted");
    {
        GestureClassifier gc;
        for (uint32_t i = 0; i < kBufferCapacity + 10; ++i)
            push_tap(gc, i * 10);

        CHECK_EQ(gc.event_count(), kBufferCapacity);
    }

    TEST_SECTION("gesture name round-trip");
    {
        for (uint8_t i = 0; i < kNumClasses; ++i) {
            const char* name = gesture_name(static_cast<Gesture>(i));
            CHECK_NE(name, nullptr);
            // Must not be empty string
            CHECK_TRUE(name[0] != '\0');
        }
    }

    TEST_SECTION("repeated classify calls are idempotent (no side effects)");
    {
        GestureClassifier gc;
        push_tap(gc, 1000);
        push_tap(gc, 1200);

        InferenceResult r1{}, r2{};
        CHECK_TRUE(gc.classify(r1));
        CHECK_TRUE(gc.classify(r2));
        CHECK_EQ(r1.gesture,    r2.gesture);
        CHECK_NEAR(r1.confidence, r2.confidence, 1e-6f);
    }

    TEST_SUMMARY();
}
