"""Generate a reproducible synthetic retail supply-chain dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path("data/raw/supply_chain_data.csv")
RANDOM_SEED = 42


def generate_supply_chain_dataset(output_path: str | Path = OUTPUT_PATH) -> str:
    """Generate and save a synthetic retail supply-chain dataset."""
    rng = np.random.default_rng(RANDOM_SEED)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    categories = ["Grocery", "Health", "Home", "Electronics", "Apparel"]
    rows: list[dict[str, object]] = []

    for idx in range(12):
        product_id = f"SKU-{idx + 1:03d}"
        category = categories[idx % len(categories)]
        base_demand = rng.integers(25, 130)
        price = round(float(rng.uniform(4.5, 95.0)), 2)
        lead_time = int(rng.integers(2, 12))
        inventory = float(base_demand * rng.uniform(9, 18))
        reorder_point = base_demand * lead_time * rng.uniform(1.2, 1.9)
        order_quantity = base_demand * rng.uniform(10, 20)
        pending_orders: list[tuple[pd.Timestamp, float]] = []

        for day_index, date in enumerate(dates):
            arriving = [qty for arrival_date, qty in pending_orders if arrival_date <= date]
            if arriving:
                inventory += sum(arriving)
                pending_orders = [(arrival_date, qty) for arrival_date, qty in pending_orders if arrival_date > date]

            weekly = 1.0 + 0.22 * np.sin(2 * np.pi * date.dayofweek / 7)
            yearly = 1.0 + 0.18 * np.sin(2 * np.pi * day_index / 365)
            monthly = 1.0 + 0.08 * np.sin(2 * np.pi * date.day / 30.5)
            promotion = int(rng.random() < (0.12 if date.dayofweek in [4, 5] else 0.07))
            promo_effect = 1.35 if promotion else 1.0
            noise = rng.normal(0, base_demand * 0.12)
            latent_demand = max(0.0, base_demand * weekly * yearly * monthly * promo_effect + noise)
            units_sold = int(min(round(latent_demand), inventory))
            inventory = max(inventory - units_sold, 0.0)

            if inventory <= reorder_point and not pending_orders:
                arrival_date = date + pd.Timedelta(days=lead_time + int(rng.integers(-1, 2)))
                pending_orders.append((arrival_date, float(order_quantity * rng.uniform(0.85, 1.2))))

            if rng.random() < 0.015:
                inventory = float(rng.uniform(0, max(1, base_demand * 1.5)))
            if rng.random() < 0.015:
                inventory += float(base_demand * rng.uniform(18, 30))

            effective_price = round(price * (0.9 if promotion else 1.0) * rng.uniform(0.98, 1.02), 2)
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "product_id": product_id,
                    "category": category,
                    "units_sold": units_sold,
                    "price": effective_price,
                    "promotion": promotion,
                    "inventory_level": round(inventory, 2),
                    "supplier_lead_time": lead_time,
                    "stockout_flag": int(inventory <= base_demand * 0.5),
                    "overstock_flag": int(inventory >= base_demand * 20),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    return output.as_posix()


if __name__ == "__main__":
    path = generate_supply_chain_dataset()
    print(f"Generated dataset: {path}")

