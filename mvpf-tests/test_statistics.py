from __future__ import annotations

from common import friedman_test, holm_adjust, rank_values, wilcoxon_signed_rank


def main() -> None:
    assert rank_values([1.0, 1.0, 3.0]) == [1.5, 1.5, 3.0]
    all_positive = wilcoxon_signed_rank([1.0, 2.0, 3.0])
    assert abs(all_positive["p_value"] - 0.25) < 1e-12
    assert abs(all_positive["rank_biserial"] - 1.0) < 1e-12
    with_zero = wilcoxon_signed_rank([0.0, 1.0, -2.0])
    assert with_zero["zeros"] == 1.0
    identical = friedman_test([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    assert abs(identical["statistic"]) < 1e-12
    assert abs(identical["p_value"] - 1.0) < 1e-12
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.20})
    assert adjusted == {"a": 0.03, "b": 0.08, "c": 0.2}
    print("statistical known-answer tests passed")


if __name__ == "__main__":
    main()
