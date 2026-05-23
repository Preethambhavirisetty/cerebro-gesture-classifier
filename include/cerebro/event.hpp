#pragma once
#include <cstddef>
#include <cstdint>

namespace cerebro {

enum class EventType : uint8_t {
    TAP         = 0,
    HOLD_START  = 1,
    HOLD_END    = 2,
    NAVIGATE    = 3,
    DROPOUT     = 4,   // sensor-gap marker: counts in timing but not in gesture stats
};

// Field order is load-bearing for the size assertion: put int16_t before uint8_t
// so the compiler does not insert 1 byte of padding between type and value.
struct Event {
    uint32_t  timestamp_ms;   // offset 0  (4 bytes)
    int16_t   value;          // offset 4  (2 bytes) — NAVIGATE: depth; HOLD_END: ms; TAP: 0
    EventType type;           // offset 6  (1 byte)
    uint8_t   pad;            // offset 7  (1 byte, explicit padding)
};

static_assert(sizeof(Event) == 8, "Event must be 8 bytes");

constexpr std::size_t kBufferCapacity = 64;
constexpr std::size_t kNumFeatures    = 14;
constexpr std::size_t kNumClasses     = 8;
constexpr uint32_t   kWindowMs       = 2000;

enum class Gesture : uint8_t {
    IDLE             = 0,
    SINGLE_TAP       = 1,
    DOUBLE_TAP       = 2,
    TRIPLE_TAP       = 3,
    SHORT_HOLD       = 4,
    LONG_HOLD        = 5,
    SHALLOW_NAVIGATE = 6,
    DEEP_NAVIGATE    = 7,
};

const char* gesture_name(Gesture g) noexcept;

} // namespace cerebro
