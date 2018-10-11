from scipy.stats import norm
import numpy as np
from typing import Mapping, Tuple


class EuropeanBSPricing:

    def __init__(
        self,
        is_call: bool,
        spot_price: float,
        strike: float,
        expiry: float,
        r: float,
        sigma: float
    ) -> None:
        self.is_call: bool = is_call
        self.spot_price: float = spot_price
        self.strike: float = strike
        self.expiry: float = expiry
        self.r: float = r
        self.sigma: float = sigma
        self.option_price: float = self.get_option_price()
        self.greeks: Mapping[str, float] = self.get_greeks()

    def get_d1_d2(self) -> Tuple[float, float]:
        sigma_sqrt = self.sigma * np.sqrt(self.expiry)
        d1 = (np.log(self.spot_price / self.strike) +
              (self.r + self.sigma ** 2 / 2.) * self.expiry) / sigma_sqrt
        d2 = d1 - sigma_sqrt
        return d1, d2

    def get_option_price(self) -> float:
        d1, d2 = self.get_d1_d2()
        if self.is_call:
            ret = self.spot_price * norm.cdf(d1) -\
                  self.strike * np.exp(-self.r * self.expiry) * norm.cdf(d2)
        else:
            ret = self.strike * np.exp(-self.r * self.expiry) * norm.cdf(-d2)\
                  - self.spot_price * norm.cdf(-d1)
        return ret

    def get_greeks(self) -> Mapping[str, float]:
        d1, d2 = self.get_d1_d2()
        sqrtt = np.sqrt(self.expiry)

        gamma = norm.pdf(d1) / (self.spot_price * self.sigma * sqrtt)
        vega = self.spot_price * sqrtt * norm.pdf(d1)
        rho_temp = -self.strike * self.expiry * np.exp(-self.r * self.expiry)
        theta_temp1 = (self.spot_price * self.sigma * norm.pdf(d1)) / (2 * sqrtt)
        theta_temp2 = self.r * self.strike * np.exp(-self.r * self.expiry)

        if self.is_call:
            delta = norm.cdf(d1)
            theta = - theta_temp1 - theta_temp2 * norm.cdf(d2)
            rho = rho_temp * norm.cdf(d2)
        else:
            delta = -norm.cdf(-d1)
            theta = - theta_temp1 + theta_temp2 * norm.cdf(-d2)
            rho = rho_temp * norm.cdf(-d2)

        return {
            "Delta": delta,
            "Gamma": gamma,
            "Theta": theta,
            "Vega": vega,
            "Rho": rho
        }


if __name__ == "__main__":
    is_call_val = False
    spot_price_val = 80.0
    strike_val = 78.0
    expiry_val = 2.0
    r_val = 0.02
    sigma_val = 0.25
    opt_obj = EuropeanBSPricing(
        is_call=is_call_val,
        spot_price=spot_price_val,
        strike=strike_val,
        expiry=expiry_val,
        r=r_val,
        sigma=sigma_val
    )
    print(opt_obj.option_price)
    print(opt_obj.greeks)
