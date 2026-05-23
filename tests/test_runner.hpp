#pragma once
#include <cstdio>
#include <cmath>

// Lightweight single-header test framework — one executable per test file.
// No dynamic allocation; suitable for bare-metal simulation targets.

namespace cerebro_test {
    static int g_pass = 0;
    static int g_fail = 0;
} // namespace cerebro_test

#define CHECK(cond) do { \
    if (cond) { \
        ++cerebro_test::g_pass; \
    } else { \
        ++cerebro_test::g_fail; \
        std::fprintf(stderr, "  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (false)

#define CHECK_EQ(a, b)     CHECK((a) == (b))
#define CHECK_NE(a, b)     CHECK((a) != (b))
#define CHECK_TRUE(x)      CHECK(static_cast<bool>(x))
#define CHECK_FALSE(x)     CHECK(!static_cast<bool>(x))
#define CHECK_NEAR(a, b, eps) CHECK(std::fabs(static_cast<float>(a) - static_cast<float>(b)) < static_cast<float>(eps))
#define CHECK_IN01(v)      CHECK((v) >= 0.0f && (v) <= 1.0f)

#define TEST_SECTION(name) std::fprintf(stdout, "\n[%s]\n", name)

#define TEST_SUMMARY() do { \
    std::fprintf(stdout, "\n%s  pass=%d  fail=%d\n", \
        (cerebro_test::g_fail == 0 ? "PASS" : "FAIL"), \
        cerebro_test::g_pass, cerebro_test::g_fail); \
    return cerebro_test::g_fail == 0 ? 0 : 1; \
} while (false)
