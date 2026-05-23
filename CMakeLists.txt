cmake_minimum_required(VERSION 3.16)
project(CEREBRO VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Treat warnings as errors; ban exceptions and RTTI (embedded constraints)
add_compile_options(-Wall -Wextra -Wpedantic -fno-exceptions -fno-rtti)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O2 -ffunction-sections -fdata-sections)
    add_link_options(-Wl,--gc-sections)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

# Cross-compile for ARM Cortex-M4 (arm-none-eabi-g++ toolchain required)
option(TARGET_CORTEX_M4 "Compile for ARM Cortex-M4 simulator" OFF)
if(TARGET_CORTEX_M4)
    add_compile_options(
        -mcpu=cortex-m4
        -mfpu=fpv4-sp-d16
        -mfloat-abi=hard
        -mthumb
    )
    add_link_options(
        -mcpu=cortex-m4
        -mfpu=fpv4-sp-d16
        -mfloat-abi=hard
        -mthumb
    )
endif()

# ── Library ──────────────────────────────────────────────────────────────────
add_library(cerebro STATIC
    src/feature_extractor.cpp
    src/inference_engine.cpp
    src/gesture_classifier.cpp
)
target_include_directories(cerebro PUBLIC include)
target_compile_features(cerebro PUBLIC cxx_std_17)

# ── Tests ────────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)
