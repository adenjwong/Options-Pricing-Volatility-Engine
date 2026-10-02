from dataclasses import replace
from math import exp, sqrt
import unittest

from options_pricing import OptionInputs, binomial_price, black_scholes, greeks, payoff


class PricingTests(unittest.TestCase):
    def setUp(self):
        self.option = OptionInputs(100, 100, 1, 0.2, 0.05)

    def test_reference_prices(self):
        self.assertAlmostEqual(black_scholes(self.option), 10.450583572185565, places=10)
        self.assertAlmostEqual(black_scholes(replace(self.option, option_type="put")),
                               5.573526022256971, places=10)

    def test_validation(self):
        for field in ("spot", "strike", "maturity", "volatility", "rate", "dividend_yield"):
            for value in (float("nan"), float("inf"), -float("inf"), True, "1", None):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(self.option, **{field: value})
        for field, value in (("spot", 0), ("strike", -1), ("maturity", -1),
                             ("volatility", -1), ("option_type", "CALL")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(self.option, **{field: value})
        replace(self.option, rate=-0.03, dividend_yield=-0.01)

    def test_payoffs_and_expiry(self):
        for spot, call, put in ((0, 0, 100), (80, 0, 20), (100, 0, 0), (120, 20, 0)):
            for kind, expected in (("call", call), ("put", put)):
                self.assertEqual(payoff(spot, 100, kind), expected)
                if spot > 0:
                    option = replace(self.option, spot=spot, maturity=0, option_type=kind)
                    self.assertEqual(black_scholes(option), expected)
                    self.assertEqual(binomial_price(option, exercise="american"), expected)
        for args in ((-1, 100), (100, 0), (100, 100, "bad")):
            with self.assertRaises(ValueError):
                payoff(*args)

    def test_zero_volatility(self):
        for rate in (-0.05, 0.05):
            for kind in ("call", "put"):
                option = replace(self.option, volatility=0, rate=rate, dividend_yield=0.02,
                                 option_type=kind)
                sign = 1 if kind == "call" else -1
                expected = max(sign * (100 * exp(-0.02) - 100 * exp(-rate)), 0)
                self.assertAlmostEqual(black_scholes(option), expected)
                self.assertAlmostEqual(binomial_price(option), expected)
        put = replace(self.option, spot=80, volatility=0, option_type="put")
        self.assertEqual(binomial_price(put, exercise="american"), 20)
        call = replace(self.option, spot=120, volatility=0, rate=0, dividend_yield=0.1)
        self.assertEqual(binomial_price(call, exercise="american"), 20)

    def test_parity_and_bounds(self):
        for spot in (50, 100, 150):
            for rate in (-0.04, 0, 0.06):
                for maturity in (0.01, 1, 5):
                    option = replace(self.option, spot=spot, rate=rate, maturity=maturity,
                                     dividend_yield=0.03)
                    call = black_scholes(option)
                    put = black_scholes(replace(option, option_type="put"))
                    s = spot * exp(-0.03 * maturity)
                    k = 100 * exp(-rate * maturity)
                    self.assertAlmostEqual(call - put, s - k, places=10)
                    self.assertGreaterEqual(call + 1e-12, max(s - k, 0))
                    self.assertLessEqual(call, s + 1e-12)
                    self.assertGreaterEqual(put + 1e-12, max(k - s, 0))
                    self.assertLessEqual(put, k + 1e-12)

    def test_greeks_finite_differences(self):
        for kind in ("call", "put"):
            for spot in (80, 100, 120):
                option = replace(self.option, spot=spot, rate=-0.01, dividend_yield=0.03,
                                 option_type=kind)
                result = greeks(option)
                for field, greek, sign in (("spot", "delta", 1), ("volatility", "vega", 1),
                                            ("maturity", "theta", -1), ("rate", "rho", 1)):
                    for h in (1e-4, 1e-5):
                        up = black_scholes(replace(option, **{field: getattr(option, field) + h}))
                        down = black_scholes(replace(option, **{field: getattr(option, field) - h}))
                        with self.subTest(kind=kind, spot=spot, greek=greek, h=h):
                            self.assertAlmostEqual(getattr(result, greek), sign * (up - down) / (2 * h),
                                                   delta=1e-5)
                h = 0.01
                gamma = (black_scholes(replace(option, spot=spot + h))
                         - 2 * black_scholes(option)
                         + black_scholes(replace(option, spot=spot - h))) / h**2
                self.assertAlmostEqual(result.gamma, gamma, delta=1e-7)

    def test_greeks_singular_inputs(self):
        for field in ("maturity", "volatility"):
            with self.assertRaisesRegex(ValueError, "positive"):
                greeks(replace(self.option, **{field: 0}))

    def test_one_step_tree(self):
        up, down = exp(0.2), exp(-0.2)
        probability = (exp(0.05) - down) / (up - down)
        expected = exp(-0.05) * probability * (100 * up - 100)
        self.assertAlmostEqual(binomial_price(self.option, 1), expected, places=11)

    def test_tree_convergence_and_exercise(self):
        for kind in ("call", "put"):
            for rate, dividend in ((0.05, 0), (-0.02, 0.03), (0.03, 0.08)):
                option = replace(self.option, option_type=kind, rate=rate, dividend_yield=dividend)
                european = binomial_price(option, 400)
                american = binomial_price(option, 400, "american")
                self.assertAlmostEqual(european, black_scholes(option), delta=0.006)
                self.assertGreaterEqual(american + 1e-10, european)
                self.assertGreaterEqual(american + 1e-10, payoff(100, 100, kind))
                if kind == "call" and dividend == 0:
                    self.assertAlmostEqual(american, european, places=10)
        put = replace(self.option, option_type="put")
        self.assertAlmostEqual(binomial_price(put, 400, "american"), 6.09, delta=0.01)

    def test_tree_validation(self):
        for steps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                binomial_price(self.option, steps)
        with self.assertRaises(ValueError):
            binomial_price(self.option, exercise="bad")
        with self.assertRaisesRegex(ValueError, "probability"):
            binomial_price(replace(self.option, rate=1), 1)
        self.assertGreater(binomial_price(replace(self.option, rate=1), 100), 0)


if __name__ == "__main__":
    unittest.main()
