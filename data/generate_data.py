#!/usr/bin/env python3
"""Simulate a synthetic, multi-vertical delivery-app order log with two randomised
retention campaigns. Fully synthetic: no real or confidential data.

Why simulate: churn and uplift need ground truth to be checkable. The simulator plants
known drivers (bad delivery experiences, single-vertical usage, price sensitivity,
tenure) and heterogeneous treatment effects, so we can test whether the models recover
them.

Outputs (written next to this file):
  orders.csv.gz        one row per order: user_id, day, vertical, order_value_aed,
                       discount_used, delivery_delay_min, rating
  users.csv            user_id, city, signup_day
  campaigns.csv        user_id, campaign_day, arm   (randomised at day 90 and day 120)
  users_latent.csv     simulator ground truth (price_sensitivity, ...). NOT used for
                       modelling - only to sanity-check that the models found real signal.

Run:  python data/generate_data.py   (seeded, ~10 s)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N_USERS = 50_000
N_DAYS = 150
CAMPAIGN_DAYS = (90, 120)
CAMPAIGN_WINDOW = 28
ARMS = ["control", "discount", "service_recovery", "cross_sell"]
VERTICALS = ["food", "grocery", "retail"]

CITIES = {  # name: (share, median delivery minutes, delay dispersion)
    "Dubai": (0.34, 24, 0.35),
    "Abu Dhabi": (0.16, 26, 0.38),
    "Riyadh": (0.22, 29, 0.45),
    "Karachi": (0.16, 33, 0.55),
    "Amman": (0.12, 28, 0.42),
}
PROMISED_MIN = 35          # an order is "late" above this
VALUE_MEAN = {"food": 55, "grocery": 80, "retail": 150}


def main(out_dir: Path = Path(__file__).parent) -> None:
    rng = np.random.default_rng(SEED)
    n = N_USERS
    uid = np.arange(n)

    # ---- latent user traits -------------------------------------------------------
    city_names = list(CITIES)
    city = rng.choice(len(city_names), size=n, p=[CITIES[c][0] for c in city_names])
    city_median = np.array([CITIES[c][1] for c in city_names])[city]
    city_sigma = np.array([CITIES[c][2] for c in city_names])[city]

    price_sens = rng.beta(2, 3, n)                      # 0 = indifferent, 1 = very
    tenure0 = rng.uniform(0, 900, n)                    # days since signup at day 0
    signup_day = -tenure0.round().astype(int)

    adopt = np.column_stack([
        rng.random(n) < 0.90,                           # food
        rng.random(n) < 0.40,                           # grocery
        rng.random(n) < 0.22,                           # retail
    ])
    none = ~adopt.any(1)
    adopt[none, 0] = True
    base_rate = np.column_stack([
        rng.gamma(2.0, 0.06, n),                        # ~0.12 food orders/day
        rng.gamma(2.0, 0.025, n),
        rng.gamma(2.0, 0.012, n),
    ])

    # ---- dynamic state ----------------------------------------------------------
    alive = np.ones(n, bool)
    frustration = np.zeros(n)
    last_order = np.full(n, -10_000)
    hazard_mult = np.ones(n)          # campaign effect on churn hazard
    rate_mult = np.ones(n)            # campaign effect on order rate
    disc_boost = np.zeros(n)          # campaign effect on discount usage
    effect_until = np.full(n, -1)

    orders = []
    campaigns = []

    for day in range(N_DAYS):
        # ---- randomised retention campaign ----------------------------------------
        if day in CAMPAIGN_DAYS:
            eligible = np.where(last_order >= day - 60)[0]
            arm = rng.integers(0, len(ARMS), eligible.size)
            campaigns.append(pd.DataFrame({"user_id": eligible, "campaign_day": day,
                                           "arm": np.array(ARMS)[arm]}))
            single = adopt[eligible].sum(1) == 1
            m = np.ones(eligible.size)
            r = np.ones(eligible.size)
            d = np.zeros(eligible.size)
            # discount: works for price-sensitive users, little for others
            k = arm == 1
            m[k] = np.exp(-4.0 * np.maximum(price_sens[eligible][k] - 0.3, 0))
            r[k] = 1.25
            d[k] = 0.5
            # service recovery: works in proportion to accumulated bad experiences
            k = arm == 2
            m[k] = np.exp(-0.6 * np.minimum(frustration[eligible][k], 4))
            frustration[eligible[k]] *= 0.4
            # cross-sell into a new vertical: works for single-vertical users
            k = arm == 3
            m[k & single] = np.exp(-1.0)
            m[k & ~single] = np.exp(-0.1)
            newly = eligible[k & single & (rng.random(eligible.size) < 0.35)]
            for u in newly:
                unused = np.where(~adopt[u])[0]
                if unused.size:
                    adopt[u, rng.choice(unused)] = True
            hazard_mult[eligible] = m
            rate_mult[eligible] = r
            disc_boost[eligible] = d
            effect_until[eligible] = day + CAMPAIGN_WINDOW

        expired = effect_until < day
        hazard_mult[expired] = 1.0
        rate_mult[expired] = 1.0
        disc_boost[expired] = 0.0

        # ---- churn and reactivation ----------------------------------------------
        tenure = tenure0 + day
        single = adopt.sum(1) == 1
        hazard = 0.0045 * np.exp(
            0.42 * np.minimum(frustration, 5)
            + 0.55 * single
            + 1.1 * (price_sens - 0.4)
            - 0.25 * np.log1p(tenure / 180)
        ) * hazard_mult
        churn_today = alive & (rng.random(n) < hazard)
        reactivate = ~alive & (rng.random(n) < 0.004 / hazard_mult)
        alive = (alive & ~churn_today) | reactivate

        # ---- orders ---------------------------------------------------------------
        p_order = 1 - np.exp(-base_rate * adopt * rate_mult[:, None])
        placed = alive[:, None] & (rng.random((n, 3)) < p_order)
        users_v, vert = np.nonzero(placed)
        if users_v.size:
            delay = city_median[users_v] * np.exp(rng.normal(0, city_sigma[users_v]))
            late = delay > PROMISED_MIN
            rating = np.clip(
                np.round(4.7 - late * rng.integers(1, 4, users_v.size)
                         + rng.normal(0, 0.5, users_v.size)), 1, 5).astype(int)
            value = np.array([VALUE_MEAN[v] for v in VERTICALS])[vert] * rng.lognormal(
                -0.08, 0.4, users_v.size)
            disc = rng.random(users_v.size) < np.clip(
                0.02 + 0.7 * price_sens[users_v] + disc_boost[users_v], 0, 0.95)
            orders.append(pd.DataFrame({
                "user_id": users_v, "day": day, "vertical": np.array(VERTICALS)[vert],
                "order_value_aed": value.round(2), "discount_used": disc.astype(int),
                "delivery_delay_min": delay.round(1), "rating": rating,
            }))
            np.add.at(frustration, users_v, late * 1.0 + (rating <= 2) * 0.8)
            last_order[users_v] = day
        frustration *= 0.93

    out_dir.mkdir(parents=True, exist_ok=True)
    orders_df = pd.concat(orders, ignore_index=True)
    orders_df.to_csv(out_dir / "orders.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"user_id": uid, "city": np.array(city_names)[city],
                  "signup_day": signup_day}).to_csv(out_dir / "users.csv", index=False)
    pd.concat(campaigns, ignore_index=True).to_csv(out_dir / "campaigns.csv", index=False)
    pd.DataFrame({"user_id": uid, "price_sensitivity": price_sens.round(4),
                  "adopted_food": adopt[:, 0].astype(int),
                  "adopted_grocery": adopt[:, 1].astype(int),
                  "adopted_retail": adopt[:, 2].astype(int)}).to_csv(
        out_dir / "users_latent.csv", index=False)
    print(f"{len(orders_df):,} orders, {n:,} users, "
          f"{sum(len(c) for c in campaigns):,} campaign assignments -> {out_dir}")


if __name__ == "__main__":
    main()
