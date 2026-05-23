#include "test_runner.hpp"
#include "cerebro/feature_extractor.hpp"

using namespace cerebro;

static void push(EventBuffer& buf, uint32_t ts, EventType t, int16_t v = 0) {
    buf.push({ts, v, t, 0});
}

int main() {
    FeatureExtractor fx;
    FeatureVec       fv;

    TEST_SECTION("empty buffer returns false");
    {
        EventBuffer buf;
        CHECK_FALSE(fx.extract(buf, fv));
        for (float f : fv) CHECK_EQ(f, 0.0f);
    }

    TEST_SECTION("all features in [0,1]");
    {
        EventBuffer buf;
        push(buf, 1000, EventType::TAP);
        push(buf, 1200, EventType::TAP);
        push(buf, 1400, EventType::TAP);
        push(buf, 1500, EventType::HOLD_START);
        push(buf, 1800, EventType::HOLD_END, 300);
        push(buf, 1850, EventType::NAVIGATE, 15);

        CHECK_TRUE(fx.extract(buf, fv));
        for (float f : fv) CHECK_IN01(f);
    }

    TEST_SECTION("single tap: only TAP_COUNT non-zero among gesture features");
    {
        EventBuffer buf;
        push(buf, 1000, EventType::TAP);

        CHECK_TRUE(fx.extract(buf, fv));
        CHECK_TRUE(fv[feat::TAP_COUNT] > 0.0f);
        CHECK_EQ(fv[feat::HOLD_COUNT], 0.0f);
        CHECK_EQ(fv[feat::NAV_COUNT],  0.0f);
        CHECK_EQ(fv[feat::MAX_HOLD_MS], 0.0f);
        CHECK_EQ(fv[feat::TOTAL_NAV_DEPTH], 0.0f);
        CHECK_EQ(fv[feat::GESTURE_DURATION], 0.0f);  // single event, no span
    }

    TEST_SECTION("events outside window are excluded");
    {
        EventBuffer buf;
        // Event well outside window (ts=0 when newest is 5000, window=2000)
        push(buf,    0, EventType::TAP);
        push(buf, 3001, EventType::TAP);  // 5000 - 3001 = 1999 < 2000 → inside
        push(buf, 5000, EventType::TAP);  // newest

        CHECK_TRUE(fx.extract(buf, fv));
        // Only the events at 3001 and 5000 are in window — tap_count = 2
        CHECK_NEAR(fv[feat::TAP_COUNT], 2.0f / norm::TAP_COUNT, 1e-5f);
    }

    TEST_SECTION("hold duration statistics");
    {
        EventBuffer buf;
        push(buf, 1000, EventType::HOLD_START);
        push(buf, 1500, EventType::HOLD_END, 500);
        push(buf, 2000, EventType::HOLD_START);
        push(buf, 3000, EventType::HOLD_END, 1000);

        CHECK_TRUE(fx.extract(buf, fv));
        // max_hold = 1000 ms → 1000/3000
        CHECK_NEAR(fv[feat::MAX_HOLD_MS],  1000.0f / norm::HOLD_MS, 1e-4f);
        // mean_hold = (500+1000)/2 = 750 ms → 750/3000
        CHECK_NEAR(fv[feat::MEAN_HOLD_MS],  750.0f / norm::HOLD_MS, 1e-4f);
    }

    TEST_SECTION("navigation depth statistics");
    {
        EventBuffer buf;
        push(buf, 1000, EventType::NAVIGATE, 10);
        push(buf, 1200, EventType::NAVIGATE, 40);
        push(buf, 1400, EventType::NAVIGATE, -20);  // negative → abs = 20

        CHECK_TRUE(fx.extract(buf, fv));
        // total_nav = 10+40+20 = 70 → 70/100
        CHECK_NEAR(fv[feat::TOTAL_NAV_DEPTH], 70.0f / norm::NAV_DEPTH, 1e-4f);
        // max_nav = 40 → 40/100
        CHECK_NEAR(fv[feat::MAX_NAV_DEPTH], 40.0f / norm::NAV_DEPTH, 1e-4f);
        // nav_count = 3 → 3/20
        CHECK_NEAR(fv[feat::NAV_COUNT], 3.0f / norm::NAV_COUNT, 1e-5f);
    }

    TEST_SECTION("gesture duration");
    {
        EventBuffer buf;
        push(buf, 1000, EventType::TAP);
        push(buf, 1500, EventType::TAP);

        CHECK_TRUE(fx.extract(buf, fv));
        // duration = 1500 - 1000 = 500 ms → 500/2000 = 0.25
        CHECK_NEAR(fv[feat::GESTURE_DURATION], 0.25f, 1e-5f);
    }

    TEST_SECTION("event rate");
    {
        EventBuffer buf;
        // 5 events over 400ms → 4 intervals / (400/1000) = 10 Hz → 10/30
        for (uint32_t i = 0; i < 5; ++i)
            push(buf, 1000 + i * 100, EventType::TAP);

        CHECK_TRUE(fx.extract(buf, fv));
        CHECK_NEAR(fv[feat::EVENT_RATE_HZ], 10.0f / norm::RATE_HZ, 1e-4f);
    }

    TEST_SECTION("tap regularity: 3+ evenly-spaced taps → near 1.0");
    {
        EventBuffer buf;
        push(buf, 1000, EventType::TAP);
        push(buf, 1200, EventType::TAP);
        push(buf, 1400, EventType::TAP);

        CHECK_TRUE(fx.extract(buf, fv));
        // Perfectly even → CV=0, regularity=1
        CHECK_NEAR(fv[feat::TAP_REGULARITY], 1.0f, 1e-4f);
    }

    TEST_SECTION("tap regularity: fewer than 3 taps → 0");
    {
        EventBuffer buf;
        push(buf, 1000, EventType::TAP);
        push(buf, 1300, EventType::TAP);

        CHECK_TRUE(fx.extract(buf, fv));
        CHECK_EQ(fv[feat::TAP_REGULARITY], 0.0f);
    }

    TEST_SECTION("window utilization caps at 1.0 with full buffer");
    {
        EventBuffer buf;
        for (uint32_t i = 0; i < kBufferCapacity; ++i)
            push(buf, 1000 + i * 10, EventType::TAP);

        CHECK_TRUE(fx.extract(buf, fv));
        CHECK_NEAR(fv[feat::WINDOW_UTILIZATION], 1.0f, 1e-5f);
    }

    TEST_SUMMARY();
}
