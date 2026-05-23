#include "test_runner.hpp"
#include "cerebro/circular_buffer.hpp"
#include "cerebro/event.hpp"

using namespace cerebro;

static Event make_tap(uint32_t ts) {
    return {ts, 0, EventType::TAP, 0};
}

int main() {
    TEST_SECTION("empty buffer");
    {
        CircularBuffer<Event, 4> cb;
        CHECK_TRUE(cb.empty());
        CHECK_FALSE(cb.full());
        CHECK_EQ(cb.size(), 0u);
        CHECK_EQ(cb.capacity(), 4u);
    }

    TEST_SECTION("push below capacity");
    {
        CircularBuffer<Event, 4> cb;
        cb.push(make_tap(10));
        cb.push(make_tap(20));
        cb.push(make_tap(30));

        CHECK_EQ(cb.size(), 3u);
        CHECK_FALSE(cb.full());
        CHECK_EQ(cb[0].timestamp_ms, 10u);  // oldest
        CHECK_EQ(cb[1].timestamp_ms, 20u);
        CHECK_EQ(cb[2].timestamp_ms, 30u);  // newest
        CHECK_EQ(cb.oldest().timestamp_ms, 10u);
        CHECK_EQ(cb.newest().timestamp_ms, 30u);
    }

    TEST_SECTION("fill to capacity");
    {
        CircularBuffer<Event, 4> cb;
        for (uint32_t i = 1; i <= 4; ++i) cb.push(make_tap(i * 10));

        CHECK_TRUE(cb.full());
        CHECK_EQ(cb.size(), 4u);
        CHECK_EQ(cb[0].timestamp_ms, 10u);
        CHECK_EQ(cb[3].timestamp_ms, 40u);
    }

    TEST_SECTION("overwrite oldest when full");
    {
        CircularBuffer<Event, 4> cb;
        for (uint32_t i = 1; i <= 4; ++i) cb.push(make_tap(i * 10));

        cb.push(make_tap(50));  // evicts ts=10

        CHECK_EQ(cb.size(), 4u);
        CHECK_EQ(cb[0].timestamp_ms, 20u);  // oldest is now 20
        CHECK_EQ(cb[3].timestamp_ms, 50u);  // newest is 50

        cb.push(make_tap(60));  // evicts ts=20
        CHECK_EQ(cb[0].timestamp_ms, 30u);
        CHECK_EQ(cb[3].timestamp_ms, 60u);
    }

    TEST_SECTION("wrap-around indexing consistency");
    {
        CircularBuffer<Event, 4> cb;
        for (uint32_t i = 1; i <= 8; ++i) cb.push(make_tap(i * 100));

        // After 8 pushes into capacity-4 buffer: contains 5,6,7,8 (×100ms)
        CHECK_EQ(cb.size(), 4u);
        CHECK_EQ(cb[0].timestamp_ms, 500u);
        CHECK_EQ(cb[1].timestamp_ms, 600u);
        CHECK_EQ(cb[2].timestamp_ms, 700u);
        CHECK_EQ(cb[3].timestamp_ms, 800u);
    }

    TEST_SECTION("clear resets state");
    {
        CircularBuffer<Event, 4> cb;
        cb.push(make_tap(1));
        cb.push(make_tap(2));
        cb.clear();

        CHECK_TRUE(cb.empty());
        CHECK_EQ(cb.size(), 0u);

        cb.push(make_tap(99));
        CHECK_EQ(cb.size(), 1u);
        CHECK_EQ(cb[0].timestamp_ms, 99u);
    }

    TEST_SECTION("mutable element access");
    {
        CircularBuffer<Event, 4> cb;
        cb.push(make_tap(1));
        cb[0].timestamp_ms = 42;
        CHECK_EQ(cb[0].timestamp_ms, 42u);
    }

    TEST_SECTION("large capacity stress");
    {
        CircularBuffer<Event, kBufferCapacity> cb;
        for (uint32_t i = 0; i < 200; ++i) cb.push(make_tap(i));

        CHECK_TRUE(cb.full());
        CHECK_EQ(cb.size(), kBufferCapacity);
        // Oldest should be 200 - 64 = 136
        CHECK_EQ(cb[0].timestamp_ms, 200u - kBufferCapacity);
        CHECK_EQ(cb[kBufferCapacity - 1].timestamp_ms, 199u);
    }

    TEST_SUMMARY();
}
