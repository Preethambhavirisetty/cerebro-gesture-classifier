"""
Feature extraction for gesture event sequences.

CRITICAL: This file must produce identical output to the C++ FeatureExtractor
in src/feature_extractor.cpp for the same input.  Any change here must be
mirrored in the C++ implementation and vice versa.
"""

import math
from typing import List, Tuple

import numpy as np

# ── Constants (must match C++ event.hpp / feature_extractor.hpp) ──────────────
K_WINDOW_MS       = 2000
K_BUFFER_CAPACITY = 64
K_NUM_FEATURES    = 14
K_NUM_CLASSES     = 8

# Event types
TAP        = 0
HOLD_START = 1
HOLD_END   = 2
NAVIGATE   = 3
DROPOUT    = 4

# Feature indices (must match C++ feat:: namespace)
FEAT_EVENT_COUNT        = 0
FEAT_TAP_COUNT          = 1
FEAT_HOLD_COUNT         = 2
FEAT_NAV_COUNT          = 3
FEAT_MEAN_IEI_MS        = 4
FEAT_STD_IEI_MS         = 5
FEAT_MAX_HOLD_MS        = 6
FEAT_MEAN_HOLD_MS       = 7
FEAT_TOTAL_NAV_DEPTH    = 8
FEAT_MAX_NAV_DEPTH      = 9
FEAT_EVENT_RATE_HZ      = 10
FEAT_WINDOW_UTILIZATION = 11
FEAT_GESTURE_DURATION   = 12
FEAT_TAP_REGULARITY     = 13

FEATURE_NAMES = [
    "event_count", "tap_count", "hold_count", "nav_count",
    "mean_iei_ms", "std_iei_ms", "max_hold_ms", "mean_hold_ms",
    "total_nav_depth", "max_nav_depth", "event_rate_hz",
    "window_utilization", "gesture_duration", "tap_regularity",
]

# Normalization divisors (must match C++ norm:: namespace)
NORM_EVENT_COUNT  = 64.0
NORM_TAP_COUNT    = 10.0
NORM_HOLD_COUNT   =  5.0
NORM_NAV_COUNT    = 20.0
NORM_IEI_MS       = 2000.0
NORM_HOLD_MS      = 3000.0
NORM_NAV_DEPTH    = 100.0
NORM_RATE_HZ      = 30.0
NORM_DURATION_MS  = 2000.0
NORM_TAP_REG      =  1.0

Event = Tuple[int, int, int]  # (timestamp_ms, event_type, value)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def extract_features(events: List[Event]) -> np.ndarray:
    """
    Extract the 14-element feature vector from an event sequence.

    Args:
        events: list of (timestamp_ms, event_type, value), any order.

    Returns:
        np.ndarray of shape (K_NUM_FEATURES,), dtype float32, values in [0, 1].

    Algorithm mirrors C++ FeatureExtractor::extract() exactly:
      1. Sort by timestamp.
      2. Keep only the most recent K_WINDOW_MS worth of events.
      3. Cap at K_BUFFER_CAPACITY (simulate circular-buffer overflow).
      4. Compute per-type statistics + timing statistics.
      5. Normalize with the same divisors and clamp to [0, 1].
    """
    features = np.zeros(K_NUM_FEATURES, dtype=np.float32)

    if not events:
        return features

    # 1. Sort oldest-first (mirrors C++ circular buffer iteration order)
    sorted_events = sorted(events, key=lambda e: e[0])

    # 2. Window filter
    newest_ts = sorted_events[-1][0]
    cutoff_ts = max(0, newest_ts - K_WINDOW_MS)
    in_window = [e for e in sorted_events if e[0] >= cutoff_ts]

    if not in_window:
        return features

    # 3. Simulate buffer capacity (keep newest K_BUFFER_CAPACITY events)
    if len(in_window) > K_BUFFER_CAPACITY:
        in_window = in_window[-K_BUFFER_CAPACITY:]

    timestamps   = [e[0] for e in in_window]
    event_count  = len(in_window)
    tap_count    = 0
    hold_count   = 0
    nav_count    = 0
    max_hold_ms  = 0.0
    sum_hold_ms  = 0.0
    total_nav    = 0.0
    max_nav      = 0.0
    tap_ts: List[int] = []

    for ts, etype, value in in_window:
        if etype == TAP:
            tap_count += 1
            tap_ts.append(ts)
        elif etype == HOLD_START:
            hold_count += 1
        elif etype == HOLD_END:
            dur = max(0.0, float(value))
            sum_hold_ms += dur
            if dur > max_hold_ms:
                max_hold_ms = dur
        elif etype == NAVIGATE:
            nav_count += 1
            depth = abs(float(value))
            total_nav += depth
            if depth > max_nav:
                max_nav = depth
        # DROPOUT: falls through — counted in timestamps but not gesture stats

    # 4a. Inter-event interval statistics (population std, ddof=0)
    mean_iei = 0.0
    std_iei  = 0.0
    if len(timestamps) > 1:
        ieis     = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        mean_iei = float(np.mean(ieis))
        std_iei  = float(np.std(ieis, ddof=0))   # population std, matches C++

    # 4b. Event rate over in-window span
    span_ms  = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 1.0
    rate_hz  = (event_count - 1) / (span_ms / 1000.0) if span_ms > 0.0 else 0.0

    # 4c. Hold statistics
    mean_hold_ms = sum_hold_ms / hold_count if hold_count > 0 else 0.0

    # 4d. Tap regularity: 1 - CV of inter-tap intervals (requires ≥ 3 taps)
    tap_reg = 0.0
    if len(tap_ts) > 2:
        tap_ieis = [tap_ts[i] - tap_ts[i - 1] for i in range(1, len(tap_ts))]
        mean_iti = float(np.mean(tap_ieis))
        std_iti  = float(np.std(tap_ieis, ddof=0))
        cv       = std_iti / mean_iti if mean_iti > 0.0 else 1.0
        tap_reg  = 1.0 - min(1.0, cv)

    # 4e. Gesture duration (oldest-to-newest span of in-window events)
    gesture_dur = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0

    # 4f. Window utilization
    win_util = event_count / K_BUFFER_CAPACITY

    # 5. Pack and normalize
    features[FEAT_EVENT_COUNT]        = _clamp01(event_count  / NORM_EVENT_COUNT)
    features[FEAT_TAP_COUNT]          = _clamp01(tap_count    / NORM_TAP_COUNT)
    features[FEAT_HOLD_COUNT]         = _clamp01(hold_count   / NORM_HOLD_COUNT)
    features[FEAT_NAV_COUNT]          = _clamp01(nav_count    / NORM_NAV_COUNT)
    features[FEAT_MEAN_IEI_MS]        = _clamp01(mean_iei     / NORM_IEI_MS)
    features[FEAT_STD_IEI_MS]         = _clamp01(std_iei      / NORM_IEI_MS)
    features[FEAT_MAX_HOLD_MS]        = _clamp01(max_hold_ms  / NORM_HOLD_MS)
    features[FEAT_MEAN_HOLD_MS]       = _clamp01(mean_hold_ms / NORM_HOLD_MS)
    features[FEAT_TOTAL_NAV_DEPTH]    = _clamp01(total_nav    / NORM_NAV_DEPTH)
    features[FEAT_MAX_NAV_DEPTH]      = _clamp01(max_nav      / NORM_NAV_DEPTH)
    features[FEAT_EVENT_RATE_HZ]      = _clamp01(rate_hz      / NORM_RATE_HZ)
    features[FEAT_WINDOW_UTILIZATION] = _clamp01(win_util)
    features[FEAT_GESTURE_DURATION]   = _clamp01(gesture_dur  / NORM_DURATION_MS)
    features[FEAT_TAP_REGULARITY]     = _clamp01(tap_reg      / NORM_TAP_REG)

    return features


def extract_batch(event_sequences: List[List[Event]]) -> np.ndarray:
    """Vectorised extraction: returns shape (N, K_NUM_FEATURES)."""
    return np.stack([extract_features(seq) for seq in event_sequences])
