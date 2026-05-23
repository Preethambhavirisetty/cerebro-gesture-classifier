#pragma once
#include <array>
#include <cstddef>

namespace cerebro {

// Stack-allocated circular buffer — zero heap usage, fixed capacity N.
// Not thread-safe by design; callers must serialize on embedded targets.
// Indexing: [0] = oldest element, [size()-1] = newest element.
template<typename T, std::size_t N>
class CircularBuffer {
public:
    CircularBuffer() noexcept : head_(0), count_(0) {}

    void push(const T& item) noexcept {
        const std::size_t slot = (head_ + count_) % N;
        buf_[slot] = item;
        if (count_ < N) {
            ++count_;
        } else {
            head_ = (head_ + 1) % N;   // evict oldest
        }
    }

    const T& operator[](std::size_t i) const noexcept {
        return buf_[(head_ + i) % N];
    }

    T& operator[](std::size_t i) noexcept {
        return buf_[(head_ + i) % N];
    }

    const T& newest() const noexcept { return (*this)[count_ - 1]; }
    const T& oldest() const noexcept { return (*this)[0]; }

    std::size_t            size()     const noexcept { return count_; }
    bool                   empty()    const noexcept { return count_ == 0; }
    bool                   full()     const noexcept { return count_ == N; }
    constexpr std::size_t  capacity() const noexcept { return N; }

    void clear() noexcept { head_ = 0; count_ = 0; }

private:
    std::array<T, N> buf_;
    std::size_t      head_;
    std::size_t      count_;
};

} // namespace cerebro
