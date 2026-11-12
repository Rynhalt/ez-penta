#!/usr/bin/env python3
"""Exact planning calculations for the preregistered one-sided sign test.

This is design analysis only. Assumed success probabilities are scenarios,
not estimates of the metric's actual performance.
"""

from __future__ import annotations

import argparse
import json
from math import comb


def binomial_upper_tail(k: int, n: int, p: float) -> float:
    if n < 0 or k < 0 or k > n or not 0.0 <= p <= 1.0:
        raise ValueError("Require n >= 0, 0 <= k <= n, and 0 <= p <= 1")
    return sum(comb(n, j) * p**j * (1.0 - p) ** (n - j) for j in range(k, n + 1))


def sign_test_operating_characteristics(n: int, assumed_success_probability: float,
                                        alpha: float = 0.05) -> dict[str, float | int]:
    if n < 1:
        raise ValueError("n must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not 0.0 <= assumed_success_probability <= 1.0:
        raise ValueError("assumed_success_probability must be between 0 and 1")
    critical_successes = next(
        (k for k in range(n + 1) if binomial_upper_tail(k, n, 0.5) <= alpha),
        n + 1,
    )
    if critical_successes == n + 1:
        return {
            "n": n,
            "critical_successes": critical_successes,
            "null_tail_probability": 0.0,
            "power": 0.0,
        }
    return {
        "n": n,
        "critical_successes": critical_successes,
        "null_tail_probability": binomial_upper_tail(critical_successes, n, 0.5),
        "power": binomial_upper_tail(critical_successes, n, assumed_success_probability),
    }


def minimum_n_for_power(assumed_success_probability: float, target_power: float = 0.8,
                        alpha: float = 0.05, max_n: int = 10000) -> dict[str, float | int]:
    if not 0.5 < assumed_success_probability <= 1.0:
        raise ValueError("assumed_success_probability must be greater than 0.5 and at most 1")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be between 0 and 1")
    for n in range(1, max_n + 1):
        result = sign_test_operating_characteristics(n, assumed_success_probability, alpha)
        if result["power"] >= target_power:
            result["assumed_success_probability"] = assumed_success_probability
            result["target_power"] = target_power
            result["alpha"] = alpha
            return result
    raise ValueError(f"Target power not attained by max_n={max_n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assumed-success-probability", type=float, nargs="+",
                        default=[0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    result = {
        "schema": "lexsi.metric_sign_test_power_scenarios/1",
        "purpose": "planning_only_not_an_empirical_metric_performance_estimate",
        "test": "one-sided exact binomial sign test; H0 success probability=0.5",
        "scenarios": [
            minimum_n_for_power(p, args.power, args.alpha)
            for p in args.assumed_success_probability
        ],
        "note": "Ignores ties and the additional absolute-sign majority criterion; actual cohort needs may be larger.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
