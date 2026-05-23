"""
Synthetic gesture event-sequence generator.

Each sample is a list of (timestamp_ms, event_type, value) tuples sorted by
timestamp.  Labels mirror the C++ Gesture enum exactly.
"""

import random
from typing import List, Tuple

# ── Event types (must match C++ EventType enum) ───────────────────────────────
TAP        = 0
HOLD_START = 1
HOLD_END   = 2
NAVIGATE   = 3
DROPOUT    = 4

# ── Gesture labels (must match C++ Gesture enum) ──────────────────────────────
IDLE             = 0
SINGLE_TAP       = 1
DOUBLE_TAP       = 2
TRIPLE_TAP       = 3
SHORT_HOLD       = 4
LONG_HOLD        = 5
SHALLOW_NAVIGATE = 6
DEEP_NAVIGATE    = 7

GESTURE_NAMES = [
    "IDLE", "SINGLE_TAP", "DOUBLE_TAP", "TRIPLE_TAP",
    "SHORT_HOLD", "LONG_HOLD", "SHALLOW_NAVIGATE", "DEEP_NAVIGATE",
]

Event = Tuple[int, int, int]   # (timestamp_ms, event_type, value)


def _jitter(ts: int, sigma: int = 15) -> int:
    return max(0, ts + random.randint(-sigma, sigma))


def _add_noise(events: List[Event], base_ts: int, rate: float = 0.15) -> List[Event]:
    """Randomly sprinkle a couple of unrelated events for realism."""
    if random.random() < rate:
        t = base_ts + random.randint(0, 500)
        events.append((t, random.choice([TAP, NAVIGATE]), random.randint(1, 5)))
    return events


def _generate_idle(base_ts: int) -> List[Event]:
    events: List[Event] = []
    n = random.randint(0, 2)
    for _ in range(n):
        t    = base_ts + random.randint(0, 1999)
        kind = random.choice([TAP, NAVIGATE, DROPOUT])
        val  = random.randint(1, 5) if kind == NAVIGATE else 0
        events.append((t, kind, val))
    return events


def _generate_single_tap(base_ts: int) -> List[Event]:
    t = base_ts + random.randint(200, 1700)
    events: List[Event] = [(_jitter(t), TAP, 0)]
    _add_noise(events, base_ts, rate=0.2)
    return events


def _generate_double_tap(base_ts: int) -> List[Event]:
    t1  = base_ts + random.randint(200, 1400)
    gap = random.randint(150, 350)
    return [(_jitter(t1), TAP, 0), (_jitter(t1 + gap), TAP, 0)]


def _generate_triple_tap(base_ts: int) -> List[Event]:
    t1  = base_ts + random.randint(200, 1000)
    gap = random.randint(130, 280)
    return [
        (_jitter(t1),          TAP, 0),
        (_jitter(t1 + gap),    TAP, 0),
        (_jitter(t1 + 2*gap),  TAP, 0),
    ]


def _generate_short_hold(base_ts: int) -> List[Event]:
    t_start = base_ts + random.randint(200, 1200)
    dur     = random.randint(100, 499)
    return [
        (t_start,         HOLD_START, 0),
        (t_start + dur,   HOLD_END,   dur),
    ]


def _generate_long_hold(base_ts: int) -> List[Event]:
    t_start = base_ts + random.randint(100, 500)
    dur     = random.randint(1000, 3000)
    # Clamp so the HOLD_END stays within the 2-second window representation
    dur = min(dur, 1999 - (t_start - base_ts))
    dur = max(dur, 1000)
    return [
        (t_start,         HOLD_START, 0),
        (t_start + dur,   HOLD_END,   dur),
    ]


def _generate_shallow_navigate(base_ts: int) -> List[Event]:
    n_nav = random.randint(1, 3)
    t     = base_ts + random.randint(100, 600)
    events: List[Event] = []
    for _ in range(n_nav):
        depth = random.randint(1, 10)
        events.append((_jitter(t), NAVIGATE, depth))
        t += random.randint(200, 600)
    return events


def _generate_deep_navigate(base_ts: int) -> List[Event]:
    events: List[Event] = []
    if random.random() < 0.5:
        # Many navigations with moderate depth
        n_nav = random.randint(4, 8)
        t     = base_ts + random.randint(50, 300)
        for _ in range(n_nav):
            depth = random.randint(5, 25)
            events.append((_jitter(t), NAVIGATE, depth))
            t += random.randint(80, 250)
    else:
        # Few navigations with large depth
        n_nav = random.randint(1, 3)
        t     = base_ts + random.randint(100, 600)
        for _ in range(n_nav):
            depth = random.randint(30, 100)
            events.append((_jitter(t), NAVIGATE, depth))
            t += random.randint(300, 700)
    return events


_GENERATORS = {
    IDLE:             _generate_idle,
    SINGLE_TAP:       _generate_single_tap,
    DOUBLE_TAP:       _generate_double_tap,
    TRIPLE_TAP:       _generate_triple_tap,
    SHORT_HOLD:       _generate_short_hold,
    LONG_HOLD:        _generate_long_hold,
    SHALLOW_NAVIGATE: _generate_shallow_navigate,
    DEEP_NAVIGATE:    _generate_deep_navigate,
}


def generate_sample(label: int, base_ts: int = 1000) -> List[Event]:
    """Generate one event sequence for the given gesture label."""
    events = _GENERATORS[label](base_ts)

    # Occasionally inject DROPOUT events to simulate sensor gaps
    if random.random() < 0.1:
        t = base_ts + random.randint(0, 1999)
        events.append((t, DROPOUT, 0))

    events.sort(key=lambda e: e[0])
    return events


def generate_dataset(
    n_per_class: int = 1200,
    seed: int = 42,
) -> Tuple[List[List[Event]], List[int]]:
    """
    Returns (all_events, all_labels), shuffled.
    all_events[i] is a List[Event]; all_labels[i] is an int in [0, 7].
    """
    random.seed(seed)

    all_events: List[List[Event]] = []
    all_labels: List[int]         = []

    for label in range(8):
        for _ in range(n_per_class):
            all_events.append(generate_sample(label))
            all_labels.append(label)

    # Shuffle in sync
    combined = list(zip(all_events, all_labels))
    random.shuffle(combined)
    all_events, all_labels = zip(*combined)  # type: ignore[assignment]
    return list(all_events), list(all_labels)
