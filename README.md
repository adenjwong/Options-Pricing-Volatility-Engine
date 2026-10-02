# Options Pricing & Volatility Engine

Python 3.10+ research library. Version 0.1.0 provides European Black–Scholes–Merton prices and Greeks with continuous dividends, terminal payoffs, and European/American CRR binomial trees. No runtime dependencies.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
options-pricing-demo
python -m unittest discover -s tests -v
```

```python
from options_pricing import OptionInputs, black_scholes, greeks, binomial_price

option = OptionInputs(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
print(black_scholes(option))
print(greeks(option))
print(binomial_price(option, steps=400, exercise="american"))
```

Spot and strike are positive and use the same currency. Maturity is in years; volatility, continuously compounded rate, and continuous dividend yield are annual decimals. Rates may be negative; option type is `call` (default) or `put`. Expiry returns intrinsic value; zero volatility uses deterministic discounted payoffs. Greeks require positive maturity and volatility: delta/gamma are per spot unit, vega/rho per unit decimal (divide by 100 for a percentage point), and theta per elapsed year (divide by 365 for calendar-day reporting).

Trees approximate exercise on a discrete time grid, including at zero volatility; invalid risk-neutral probabilities require more steps. Runtime is quadratic in steps. Models assume constant parameters and are intended for research. Extreme inputs can exceed floating-point range. Implied volatility, market data, surfaces, and portfolio analysis are deferred.
