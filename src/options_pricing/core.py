from dataclasses import dataclass
from math import erfc, exp, isfinite, log, pi, sqrt
from numbers import Real


def _number(name: str, value: float, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _kind(option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")


@dataclass(frozen=True)
class OptionInputs:
    spot: float
    strike: float
    maturity: float
    volatility: float
    rate: float = 0.0
    dividend_yield: float = 0.0
    option_type: str = "call"

    def __post_init__(self) -> None:
        for name in ("spot", "strike", "maturity", "volatility", "rate", "dividend_yield"):
            _number(name, getattr(self, name))
        if self.spot <= 0 or self.strike <= 0:
            raise ValueError("spot and strike must be positive")
        if self.maturity < 0 or self.volatility < 0:
            raise ValueError("maturity and volatility must be nonnegative")
        _kind(self.option_type)


def payoff(spot: float, strike: float, option_type: str = "call") -> float:
    _number("spot", spot, 0)
    _number("strike", strike)
    if strike <= 0:
        raise ValueError("strike must be positive")
    _kind(option_type)
    return max(spot - strike if option_type == "call" else strike - spot, 0.0)


def _cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2.0))


def _terms(option: OptionInputs) -> tuple[float, float, float, float]:
    root_t = sqrt(option.maturity)
    d1 = ((log(option.spot) - log(option.strike)) / (option.volatility * root_t)
          + (option.rate - option.dividend_yield) * root_t / option.volatility
          + 0.5 * option.volatility * root_t)
    d2 = d1 - option.volatility * root_t
    return d1, d2, exp(-option.dividend_yield * option.maturity), exp(-option.rate * option.maturity)


def black_scholes(option: OptionInputs) -> float:
    if option.maturity == 0:
        return payoff(option.spot, option.strike, option.option_type)
    if option.volatility == 0:
        return payoff(option.spot * exp(-option.dividend_yield * option.maturity),
                      option.strike * exp(-option.rate * option.maturity), option.option_type)
    d1, d2, dividend_discount, discount = _terms(option)
    sign = 1 if option.option_type == "call" else -1
    return max(sign * (option.spot * dividend_discount * _cdf(sign * d1)
                       - option.strike * discount * _cdf(sign * d2)), 0.0)


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def greeks(option: OptionInputs) -> Greeks:
    # Derivatives use unit decimal volatility/rates and annual calendar-time theta.
    if option.maturity == 0 or option.volatility == 0:
        raise ValueError("Greeks require positive maturity and volatility")
    d1, d2, dividend_discount, discount = _terms(option)
    sign = 1 if option.option_type == "call" else -1
    density = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)
    root_t = sqrt(option.maturity)
    delta = sign * dividend_discount * _cdf(sign * d1)
    gamma = dividend_discount * density / (option.spot * option.volatility * root_t)
    vega = option.spot * dividend_discount * density * root_t
    theta = (-option.spot * dividend_discount * density * option.volatility / (2 * root_t)
             - sign * option.rate * option.strike * discount * _cdf(sign * d2)
             + sign * option.dividend_yield * option.spot * dividend_discount * _cdf(sign * d1))
    rho = sign * option.strike * option.maturity * discount * _cdf(sign * d2)
    return Greeks(delta, gamma, vega, theta, rho)
