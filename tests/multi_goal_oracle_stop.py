"""Lightweight checks for current-goal oracle stopping."""

from longnav.env.habitat_multi import MultiObjectHabitatWorker


def make_worker(distance, *, fn_guard=True, fp_guard=False):
    worker = object.__new__(MultiObjectHabitatWorker)
    worker._multi = {"goals": ["chair", "plant", "bed"], "goal_idx": 0}
    worker.fn_guard = fn_guard
    worker.fp_guard = fp_guard
    worker._geodesic_to_goal = lambda category: distance
    worker._success_dist = lambda: 1.0
    return worker


def main():
    worker = make_worker(0.5)
    action, extras = worker._apply_stop_guards(1)
    assert action == 0
    assert extras["+fn_stop"] == 1

    worker._fabricated_stop_step = lambda supplementary_logs={}, guard_extras=None: guard_extras
    extras = worker.step(1)
    assert worker._pending == "success_advance"
    assert extras["+fn_stop"] == 1

    worker = make_worker(1.0)
    action, extras = worker._apply_stop_guards(1)
    assert action == 1
    assert extras["+fn_stop"] == 0

    worker = make_worker(0.5, fn_guard=False)
    action, extras = worker._apply_stop_guards(1)
    assert action == 1
    assert extras["+fn_stop"] == 1

    worker = make_worker(2.0, fp_guard=True)
    action, extras = worker._apply_stop_guards(0)
    assert action in {1, 2, 3}
    assert extras["+fp_stop"] == 1

    print("OK: current-goal oracle stop")


if __name__ == "__main__":
    main()
