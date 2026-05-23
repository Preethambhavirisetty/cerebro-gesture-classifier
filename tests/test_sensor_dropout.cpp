#include "test_runner.hpp"
#include "cerebro/feature_extractor.hpp"
#include "cerebro/gesture_classifier.hpp"

using namespace cerebro;

static void push(EventBuffer& b, uint32_t ts, EventType t, int16_t v = 0) {
    b.push({ts, v, t, 0});
}

int main() {
    FeatureExtractor fx;
    FeatureVec       fv;

    TEST_SECTION("DROPOUT events do not count as taps/holds/nav");
    {
        EventBuffer buf;
        push(buf, 1000, EventType::DROPOUT);
        push(buf, 1100, EventType::DROPOUT);
        push(buf, 1200, EventType::DROPOUT);

        // Buffer is non-empty, extractor should succeed (DROPOUTs add timestamps)
        CHECK_TRUE(fx.extract(buf, fv));
        CHECK_EQ(fv[feat::TAP_COUNT],       0.0f);
        CHECK_EQ(fv[feat::HOLD_COUNT],      0.0f);
        CHECK_EQ(fv[feat::NAV_COUNT],       0.0f);
        CHECK_EQ(fv[feat::MAX_HOLD_MS],     0.0f);
        CHECK_EQ(fv[feat::TOTAL_NAV_DEPTH], 0.0f);
        // Event count IS non-zero (DROPOUTs contribute to timing)
        CHECK_TRUE(fv[feat::EVENT_COUNT] > 0.0f);
    }

    TEST_SECTION("DROPOUT mixed with TAP: tap features unaffected");
    {
        EventBuffer buf_clean, buf_noisy;

        // Clean: two taps
        push(buf_clean, 1000, EventType::TAP);
        push(buf_clean, 1300, EventType::TAP);

        // Noisy: same taps plus DROPOUTs between them
        push(buf_noisy, 1000, EventType::TAP);
        push(buf_noisy, 1100, EventType::DROPOUT);
        push(buf_noisy, 1200, EventType::DROPOUT);
        push(buf_noisy, 1300, EventType::TAP);

        FeatureVec clean_fv, noisy_fv;
        CHECK_TRUE(fx.extract(buf_clean, clean_fv));
        CHECK_TRUE(fx.extract(buf_noisy, noisy_fv));

        // tap_count must be identical
        CHECK_EQ(clean_fv[feat::TAP_COUNT], noisy_fv[feat::TAP_COUNT]);
        // hold and nav must be zero in both
        CHECK_EQ(noisy_fv[feat::HOLD_COUNT], 0.0f);
        CHECK_EQ(noisy_fv[feat::NAV_COUNT],  0.0f);
    }

    TEST_SECTION("large timestamp gap: old events fall outside window");
    {
        EventBuffer buf;
        // Events from long ago (outside 2000ms window relative to newest)
        push(buf,     0, EventType::TAP);
        push(buf,   500, EventType::TAP);
        // Gap of 3000ms — old events are now outside window
        push(buf, 3500, EventType::TAP);  // newest; window = [1500, 3500]
        push(buf, 3600, EventType::TAP);

        CHECK_TRUE(fx.extract(buf, fv));
        // Only events at ts=3500 and ts=3600 are in window → tap_count = 2
        CHECK_NEAR(fv[feat::TAP_COUNT], 2.0f / norm::TAP_COUNT, 1e-5f);
    }

    TEST_SECTION("DROPOUT followed by valid events: classifier succeeds");
    {
        GestureClassifier gc;
        for (int i = 0; i < 5; ++i)
            gc.push_event({static_cast<uint32_t>(1000 + i * 50), 0, EventType::DROPOUT, 0});

        gc.push_event({1400, 0, EventType::TAP, 0});
        gc.push_event({1600, 0, EventType::TAP, 0});

        InferenceResult result{};
        CHECK_TRUE(gc.classify(result));
        CHECK_IN01(result.confidence);
    }

    TEST_SECTION("all DROPOUT buffer: classify returns true (timing events present)");
    {
        GestureClassifier gc;
        for (uint32_t i = 0; i < 10; ++i)
            gc.push_event({1000 + i * 100, 0, EventType::DROPOUT, 0});

        InferenceResult result{};
        CHECK_TRUE(gc.classify(result));
        // All gesture-type features must be zero
        FeatureVec fv2;
        fx.extract(gc.buffer(), fv2);
        CHECK_EQ(fv2[feat::TAP_COUNT],       0.0f);
        CHECK_EQ(fv2[feat::HOLD_COUNT],      0.0f);
        CHECK_EQ(fv2[feat::NAV_COUNT],       0.0f);
        CHECK_EQ(fv2[feat::TOTAL_NAV_DEPTH], 0.0f);
    }

    TEST_SECTION("boundary: single event at ts=0");
    {
        EventBuffer buf;
        push(buf, 0, EventType::TAP);
        CHECK_TRUE(fx.extract(buf, fv));
        for (float f : fv) CHECK_IN01(f);
    }

    TEST_SECTION("boundary: max timestamp (no wraparound)");
    {
        EventBuffer buf;
        const uint32_t ts = 0xFFFFF000u;
        push(buf, ts,        EventType::TAP);
        push(buf, ts + 500,  EventType::TAP);
        push(buf, ts + 1000, EventType::TAP);

        CHECK_TRUE(fx.extract(buf, fv));
        for (float f : fv) CHECK_IN01(f);
        CHECK_NEAR(fv[feat::TAP_COUNT], 3.0f / norm::TAP_COUNT, 1e-5f);
    }

    TEST_SECTION("HOLD_END with value=0 (zero-duration hold)");
    {
        EventBuffer buf;
        push(buf, 1000, EventType::HOLD_START);
        push(buf, 1000, EventType::HOLD_END, 0);

        CHECK_TRUE(fx.extract(buf, fv));
        CHECK_EQ(fv[feat::MAX_HOLD_MS],  0.0f);
        CHECK_EQ(fv[feat::MEAN_HOLD_MS], 0.0f);
        // hold_count = 1 (HOLD_START)
        CHECK_NEAR(fv[feat::HOLD_COUNT], 1.0f / norm::HOLD_COUNT, 1e-5f);
    }

    TEST_SECTION("HOLD_END with negative value: clamped to 0 duration");
    {
        EventBuffer buf;
        push(buf, 1000, EventType::HOLD_START);
        push(buf, 1050, EventType::HOLD_END, -200);  // malformed event

        CHECK_TRUE(fx.extract(buf, fv));
        CHECK_EQ(fv[feat::MAX_HOLD_MS],  0.0f);
        CHECK_EQ(fv[feat::MEAN_HOLD_MS], 0.0f);
    }

    TEST_SUMMARY();
}
