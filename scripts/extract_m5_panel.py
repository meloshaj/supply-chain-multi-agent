"""Extract and standardize 12-product 365-day panel from Walmart M5 data."""

import os
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_RAW_PATH = Path("data/raw/supply_chain_data.csv")

# 12 selected SKUs across 6 departments in Store CA_1
SELECTED_SKUS = {
    "FOODS_1": ["FOODS_1_085", "FOODS_1_218"],
    "FOODS_2": ["FOODS_2_019", "FOODS_2_244"],
    "FOODS_3": ["FOODS_3_586", "FOODS_3_064"],
    "HOBBIES_1": ["HOBBIES_1_048", "HOBBIES_1_155"],
    "HOBBIES_2": ["HOBBIES_2_043", "HOBBIES_2_033"],
    "HOUSEHOLD_1": ["HOUSEHOLD_1_418", "HOUSEHOLD_1_535"]
}

LEAD_TIMES = {
    "FOODS_1": 3,
    "FOODS_2": 4,
    "FOODS_3": 5,
    "HOBBIES_1": 9,
    "HOBBIES_2": 8,
    "HOUSEHOLD_1": 6
}

def extract_panel():
    print("Loading M5 files...")
    cal = pd.read_csv("data/calendar.csv")
    sales = pd.read_csv("data/sales_train_evaluation.csv")
    prices = pd.read_csv("data/sell_prices.csv")
    
    # Filter store CA_1
    sales_ca1 = sales[sales["store_id"] == "CA_1"].copy()
    prices_ca1 = prices[prices["store_id"] == "CA_1"].copy()
    
    # Temporal window: d_1577 to d_1941 (365 days)
    day_cols = [f"d_{i}" for i in range(1577, 1942)]
    assert len(day_cols) == 365
    
    cal_365 = cal[cal["d"].isin(day_cols)].copy()
    cal_365["date"] = pd.to_datetime(cal_365["date"])
    # Promotion indicator: event_name_1 not null or snap_CA == 1
    cal_365["promotion"] = ((cal_365["event_name_1"].notna()) | (cal_365["snap_CA"] == 1)).astype(int)
    
    rows = []
    
    for dept, items in SELECTED_SKUS.items():
        lt = LEAD_TIMES[dept]
        for item in items:
            sales_row = sales_ca1[sales_ca1["item_id"] == item].iloc[0]
            daily_units = sales_row[day_cols].values.astype(float)
            
            # Get weekly sell price and merge to daily calendar
            item_prices = prices_ca1[prices_ca1["item_id"] == item]
            merged_cal = cal_365.merge(item_prices[["wm_yr_wk", "sell_price"]], on="wm_yr_wk", how="left")
            daily_prices = merged_cal["sell_price"].ffill().bfill().values.astype(float)
            daily_promo = merged_cal["promotion"].values.astype(int)
            dates = merged_cal["date"].dt.strftime("%Y-%m-%d").values
            
            # Deterministic inventory replenishment simulation
            # I_t = max(0, I_{t-1} + Q_t - Sales_t)
            # Reorder point ROP = LTD + SS = mu * LT + z * sigma * sqrt(LT)
            mu = daily_units.mean()
            sigma = daily_units.std(ddof=1)
            z = 1.65
            ltd = mu * lt
            ss = z * sigma * np.sqrt(lt)
            rop = ltd + ss
            order_qty = max(rop * 1.2, mu * 14)
            
            inv = rop + order_qty * 0.7
            pending_orders = [] # list of (arrival_day, qty)
            
            for t in range(365):
                arriving_qty = sum(qty for arr_day, qty in pending_orders if arr_day == t)
                pending_orders = [(arr_day, qty) for arr_day, qty in pending_orders if arr_day > t]
                inv += arriving_qty
                
                sales_t = daily_units[t]
                inv = max(0.0, inv - sales_t)
                
                in_transit = sum(qty for _, qty in pending_orders)
                if inv + in_transit <= rop:
                    pending_orders.append((t + lt, order_qty))
                
                rows.append({
                    "date": dates[t],
                    "product_id": item,
                    "category": dept,
                    "units_sold": int(sales_t),
                    "price": round(float(daily_prices[t]), 2),
                    "promotion": int(daily_promo[t]),
                    "inventory_level": round(float(inv), 2),
                    "supplier_lead_time": int(lt),
                    "stockout_flag": int(inv <= mu * 0.5),
                    "overstock_flag": int(inv >= rop * 2.5),
                })
                
    df = pd.DataFrame(rows)
    # Sort deterministically
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    
    OUTPUT_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_RAW_PATH, index=False)
    print(f"Extracted standardized panel saved to {OUTPUT_RAW_PATH}")
    print(f"Shape: {df.shape} (Expected 4380, {len(df.columns)})")
    print(f"Products: {df['product_id'].nunique()} ({list(df['product_id'].unique())})")
    print(f"Categories: {df['category'].nunique()} ({list(df['category'].unique())})")
    print(f"Dates range: {df['date'].min()} to {df['date'].max()} ({df['date'].nunique()} days)")
    return df

if __name__ == "__main__":
    extract_panel()
