from dataclasses import replace

from . import OptionInputs, binomial_price, black_scholes, greeks, payoff


def main() -> None:
    option = OptionInputs(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
    print("Spot=100; maturity=1 year; volatility=20%; rate=5%; dividend yield=0%")
    print("Strike  Call payoff  Put payoff  European call  European put")
    for strike in (90, 100, 110):
        call = replace(option, strike=strike)
        put = replace(call, option_type="put")
        print(f"{strike:6}  {payoff(100, strike):11.2f}  {payoff(100, strike, 'put'):10.2f}"
              f"  {black_scholes(call):13.4f}  {black_scholes(put):12.4f}")
    print("\nATM call Greeks (vega/rho per unit decimal; theta per year):")
    print(greeks(option))
    print("\nSteps  European call  Error vs analytic  American put")
    for steps in (25, 100, 400):
        value = binomial_price(option, steps)
        american = binomial_price(replace(option, option_type="put"), steps, "american")
        print(f"{steps:5}  {value:13.6f}  {value - black_scholes(option):17.6f}  {american:12.6f}")


if __name__ == "__main__":
    main()
