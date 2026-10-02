from math import exp, expm1, isfinite, log, sqrt

from .core import OptionInputs, black_scholes, payoff


def binomial_price(option: OptionInputs, steps: int = 200, exercise: str = "european") -> float:
    # CRR backward induction uses O(steps) memory and O(steps squared) work.
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if exercise not in ("european", "american"):
        raise ValueError("exercise must be 'european' or 'american'")
    if option.maturity == 0:
        return payoff(option.spot, option.strike, option.option_type)
    dt = option.maturity / steps
    if option.volatility == 0:
        if exercise == "european":
            return black_scholes(option)
        return max(
            payoff(option.spot * exp(-option.dividend_yield * i * dt),
                   option.strike * exp(-option.rate * i * dt), option.option_type)
            for i in range(steps + 1)
        )
    jump = option.volatility * sqrt(dt)
    drift = (option.rate - option.dividend_yield) * dt
    if jump == 0 or not -jump <= drift <= jump:
        raise ValueError("Invalid risk-neutral probability; increase steps or use the analytic model")
    sign = 1 if option.option_type == "call" else -1
    log_spot = log(option.spot)
    try:
        probability = expm1(drift + jump) / expm1(2 * jump)
        discount = exp(-option.rate * dt)
        values = [max(sign * (exp(log_spot + (2 * j - steps) * jump) - option.strike), 0.0)
                  for j in range(steps + 1)]
        for level in range(steps - 1, -1, -1):
            for j in range(level + 1):
                value = discount * ((1 - probability) * values[j] + probability * values[j + 1])
                if exercise == "american":
                    intrinsic = sign * (exp(log_spot + (2 * j - level) * jump) - option.strike)
                    value = max(value, intrinsic)
                values[j] = value
    except OverflowError as error:
        raise ValueError("Tree node values exceed floating-point range") from error
    if not isfinite(values[0]):
        raise ValueError("Tree price exceeds floating-point range")
    return values[0]
