# SPDX-License-Identifier: AGPL-3.0-only
import random


def sample_fixed(items, seed, k):
    rng = random.Random(seed)
    if len(items) <= k:
        return items
    return rng.sample(items, k)


def test_sampling_seed_is_stable():
    items = list(range(100))
    a = sample_fixed(items, 20260920, 40)
    b = sample_fixed(items, 20260920, 40)
    assert a == b


def test_sampling_differs_with_other_seed():
    items = list(range(100))
    a = sample_fixed(items, 20260920, 40)
    b = sample_fixed(items, 20260921, 40)
    assert a != b
