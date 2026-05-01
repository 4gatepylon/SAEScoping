"""DO NOT SUBMIT — scratch tests for _grid_oom.py. Do not commit."""

import math

from grid_oom import GridSearcher


def _counted_fn(true_fn):
    calls = []

    def fn(bsz, seq, device):
        calls.append((bsz, seq))
        return true_fn(bsz, seq)

    return fn, calls


def _frontier(max_product):
    return lambda b, s: "ok" if b * s <= max_product else "oom"


def test_correctness_against_synthetic_frontier():
    bsz_list = [1, 2, 4, 8, 16, 32]
    seq_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    truth = _frontier(256)
    fn, calls = _counted_fn(truth)

    searcher = GridSearcher(bsz_list, seq_list, ["cpu"], fn, label="test", verbose=False)
    results = searcher.run()

    for b in bsz_list:
        for s in seq_list:
            expected = truth(b, s)
            assert results[(b, s)] == expected, f"mismatch at ({b},{s}): got {results[(b, s)]}, expected {expected}"

    full = len(bsz_list) * len(seq_list)
    upper = len(bsz_list) * math.ceil(math.log2(len(seq_list) + 1))
    assert len(calls) <= upper, f"too many tests: {len(calls)} > log-bound {upper}"
    print(f"PASS frontier=256: {len(calls)}/{full} cells tested (log-bound {upper})")


def test_all_ok_propagates_down_left():
    bsz_list = [1, 2, 4]
    seq_list = [1, 2, 4]
    fn, calls = _counted_fn(lambda b, s: "ok")
    searcher = GridSearcher(bsz_list, seq_list, ["cpu"], fn, label="test", verbose=False)
    results = searcher.run()
    for b in bsz_list:
        for s in seq_list:
            assert results[(b, s)] == "ok"
    print(f"PASS all-ok 3x3: {len(calls)} tests")


def test_all_oom_propagates_up_right():
    bsz_list = [1, 2, 4]
    seq_list = [1, 2, 4]
    fn, calls = _counted_fn(lambda b, s: "oom")
    searcher = GridSearcher(bsz_list, seq_list, ["cpu"], fn, label="test", verbose=False)
    results = searcher.run()
    for b in bsz_list:
        for s in seq_list:
            assert results[(b, s)] == "oom"
    assert len(calls) <= 3, f"all-oom should prune aggressively, got {len(calls)} tests"
    print(f"PASS all-oom 3x3: {len(calls)} tests")


def test_threshold_at_minimum():
    bsz_list = [1, 2, 4]
    seq_list = [1, 2, 4, 8]
    truth = _frontier(1)
    fn, calls = _counted_fn(truth)
    searcher = GridSearcher(bsz_list, seq_list, ["cpu"], fn, label="test", verbose=False)
    results = searcher.run()
    for b in bsz_list:
        for s in seq_list:
            assert results[(b, s)] == truth(b, s), f"mismatch at ({b},{s})"
    print(f"PASS edge-min: {len(calls)} tests")


def test_multidevice_does_not_corrupt_results():
    bsz_list = [1, 2, 4, 8]
    seq_list = [1, 2, 4, 8, 16, 32]
    truth = _frontier(16)
    fn, calls = _counted_fn(truth)
    searcher = GridSearcher(bsz_list, seq_list, ["dev0", "dev1", "dev2", "dev3"], fn, label="test", verbose=False)
    results = searcher.run()
    for b in bsz_list:
        for s in seq_list:
            assert results[(b, s)] == truth(b, s), f"mismatch at ({b},{s}): got {results[(b, s)]}"
    print(f"PASS multidevice: {len(calls)} tests")


if __name__ == "__main__":
    test_correctness_against_synthetic_frontier()
    test_all_ok_propagates_down_left()
    test_all_oom_propagates_up_right()
    test_threshold_at_minimum()
    test_multidevice_does_not_corrupt_results()
    print("\nAll tests passed.")
