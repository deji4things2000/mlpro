import math
import statistics


def social_welfare(rewards: dict[str, float]) -> float:
    return float(sum(rewards.values()))


def reward_sd(values: list[float]) -> float:
    return float(statistics.stdev(values))


def correlation(x_values: list[float], y_values: list[float]) -> float:
    mean_x = statistics.fmean(x_values)
    mean_y = statistics.fmean(y_values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in x_values) * sum((y - mean_y) ** 2 for y in y_values)
    )
    return numerator / denominator


def cooperation_rate_public_goods(step_infos: list[dict[str, object]], endowment: int) -> float:
    cooperative = 0
    total = 0
    for info in step_infos:
        actions = dict(info["actions"])
        threshold = info.get("threshold")
        fair_share = (
            max(1, int(math.ceil(float(threshold) / max(1, len(actions)))))
            if threshold is not None
            else max(1, endowment // 2)
        )
        for contribution in actions.values():
            total += 1
            if float(contribution) >= fair_share:
                cooperative += 1
    return cooperative / total if total else 0.0


def negotiation_fairness_rate(step_infos: list[dict[str, object]], total_resource: int) -> float:
    fair = 0
    total = 0
    for info in step_infos:
        demands = dict(info["demands"])
        fair_share = total_resource / max(1, len(demands))
        for demand in demands.values():
            total += 1
            if float(demand) <= fair_share:
                fair += 1
    return fair / total if total else 0.0


def fidelity_gap(sim_rates: list[float], human_rates: list[float]) -> float:
    if len(sim_rates) == 0:
        raise ValueError("sim_rates cannot be empty")
    if len(sim_rates) != len(human_rates):
        raise ValueError("sim_rates and human_rates must have equal length")
    return sum((sim - human) ** 2 for sim, human in zip(sim_rates, human_rates)) / len(sim_rates)


def bias_corrected_score(p_hat: float, q1: float, q0: float) -> float:
    denominator = q1 + q0 - 1
    if denominator <= 0:
        raise ValueError("q1 + q0 must be > 1.")
    theta = (p_hat - (1 - q0)) / denominator
    return float(max(0.0, min(1.0, theta)))


def bias_corrected_ci(
    p_hat: float,
    q1: float,
    q0: float,
    n_test: int,
    n_calib_pos: int,
    n_calib_neg: int,
    z: float = 1.96,
    var_p_override: float | None = None
) -> tuple[float, float]:
    denominator = q1 + q0 - 1
    if denominator <= 0:
        raise ValueError("q1 + q0 must be > 1.")

    theta = bias_corrected_score(p_hat, q1, q0)
    var_p = var_p_override if var_p_override is not None else (p_hat * (1 - p_hat)) / max(1, n_test)
    var_q1 = (q1 * (1 - q1)) / max(1, n_calib_pos)
    var_q0 = (q0 * (1 - q0)) / max(1, n_calib_neg)

    d_theta_dp = 1 / denominator
    d_theta_dq1 = -(p_hat - (1 - q0)) / (denominator**2)
    d_theta_dq0 = (q1 - p_hat) / (denominator**2)

    variance = (
        (d_theta_dp**2) * var_p + (d_theta_dq1**2) * var_q1 + (d_theta_dq0**2) * var_q0
    )
    standard_error = math.sqrt(max(0.0, variance))
    lower = max(0.0, theta - z * standard_error)
    upper = min(1.0, theta + z * standard_error)
    return float(lower), float(upper)
