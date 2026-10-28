import pandas as pd
import numpy as np
import joblib

raw = pd.read_csv('data/raw/supply_chain_data.csv')
train = pd.read_csv('data/processed/train_supply_chain_data.csv')
test = pd.read_csv('data/processed/test_supply_chain_data.csv')
forecasts = pd.read_csv('outputs/tables/forecast_results.csv')
risks = pd.read_csv('outputs/tables/inventory_risks.csv')
recs = pd.read_csv('outputs/recommendations/inventory_recommendations.csv')
ablations = pd.read_csv('outputs/tables/ablation_results.csv')
prod_metrics = pd.read_csv('outputs/tables/product_test_metrics.csv')

print("="*60)
print("SECTION A: QUICK REFERENCE DATES & METRICS")
print("="*60)
print(f"Raw Date Range: {raw['date'].min()} to {raw['date'].max()} (Total days: {raw['date'].nunique()}, Total rows: {len(raw)})")
print(f"Train Date Range: {train['date'].min()} to {train['date'].max()} (Days: {train['date'].nunique()}, Rows: {len(train)})")
print(f"Test Date Range: {test['date'].min()} to {test['date'].max()} (Days: {test['date'].nunique()}, Rows: {len(test)})")

print("\n" + "="*60)
print("SECTION B: TABLE 2 DESCRIPTIVE STATISTICS (N = 4,380)")
print("="*60)
for col, name in [('units_sold', 'Units sold'), ('price', 'Price ($)'), ('inventory_level', 'Inventory level (units)'), ('supplier_lead_time', 'Supplier lead time (days)')]:
    s = raw[col]
    print(f"| **{name}** | {s.mean():.2f} | {s.std():.2f} | {s.min():.2f} | {s.median():.2f} | {s.max():.2f} |")

print("\n--- PER-SKU DESCRIPTIVE SUMMARY (365-day panel & 72-day test) ---")
sku_stats = []
for pid in sorted(raw['product_id'].unique()):
    r_sku = raw[raw['product_id'] == pid]
    t_sku = test[test['product_id'] == pid]
    cat = r_sku['category'].iloc[0]
    lt = r_sku['supplier_lead_time'].iloc[0]
    price = r_sku['price'].iloc[0]
    
    # 365-day stats
    u_365_mean = r_sku['units_sold'].mean()
    u_365_std = r_sku['units_sold'].std()
    u_365_min = r_sku['units_sold'].min()
    u_365_max = r_sku['units_sold'].max()
    
    # 72-day test stats
    u_test_mean = t_sku['units_sold'].mean()
    u_test_std = t_sku['units_sold'].std()
    u_test_min = t_sku['units_sold'].min()
    u_test_max = t_sku['units_sold'].max()
    
    sku_stats.append({
        'product_id': pid,
        'category': cat,
        'lead_time': lt,
        'price': price,
        'panel_mean': u_365_mean,
        'panel_std': u_365_std,
        'panel_min': u_365_min,
        'panel_max': u_365_max,
        'test_mean': u_test_mean,
        'test_std': u_test_std,
        'test_min': u_test_min,
        'test_max': u_test_max,
    })
df_sku_stats = pd.DataFrame(sku_stats)
for idx, r in df_sku_stats.iterrows():
    print(f"| `{r['product_id']}` | {r['category']} | {r['lead_time']}d | ${r['price']:.2f} | {r['panel_mean']:.2f} | {r['panel_std']:.2f} | {r['panel_min']:.0f} - {r['panel_max']:.0f} | {r['test_mean']:.2f} | {r['test_std']:.2f} | {r['test_min']:.0f} - {r['test_max']:.0f} |")

print("\n" + "="*60)
print("SECTION C: TABLE 4 TEST SET ACCURACY (N = 864)")
print("="*60)
y = forecasts['actual_units_sold']
b = forecasts['baseline_prediction']
rf = forecasts['random_forest_prediction']
nz = y > 0

b_mae = np.mean(np.abs(y - b))
b_rmse = np.sqrt(np.mean((y - b)**2))
b_mape = np.mean(np.abs((y[nz] - b[nz]) / y[nz])) * 100
b_wape = np.sum(np.abs(y - b)) / np.sum(y) * 100

rf_mae = np.mean(np.abs(y - rf))
rf_rmse = np.sqrt(np.mean((y - rf)**2))
rf_mape = np.mean(np.abs((y[nz] - rf[nz]) / y[nz])) * 100
rf_wape = np.sum(np.abs(y - rf)) / np.sum(y) * 100

red_mae = (b_mae - rf_mae) / b_mae * 100
red_rmse = (b_rmse - rf_rmse) / b_rmse * 100
red_mape = (b_mape - rf_mape) / b_mape * 100
red_wape = (b_wape - rf_wape) / b_wape * 100

print(f"Baseline: MAE={b_mae:.2f}, RMSE={b_rmse:.2f}, MAPE={b_mape:.2f}%, WAPE={b_wape:.2f}%")
print(f"RF:       MAE={rf_mae:.2f}, RMSE={rf_rmse:.2f}, MAPE={rf_mape:.2f}%, WAPE={rf_wape:.2f}%")
print(f"Reduction: MAE={red_mae:.2f}%, RMSE={red_rmse:.2f}%, MAPE={red_mape:.2f}%, WAPE={red_wape:.2f}%")

print("\n" + "="*60)
print("SECTION D: TABLE 5 ABLATION RESULTS")
print("="*60)
full_row = ablations[ablations['configuration'].str.contains('Full model')].iloc[0]
full_mae = full_row['mae']
full_rmse = full_row['rmse']

for idx, r in ablations.iterrows():
    delta_mae = (r['mae'] - full_mae) / full_mae * 100
    delta_rmse = (r['rmse'] - full_rmse) / full_rmse * 100
    print(f"| **{r['configuration']}** | {r['mae']:.2f} | {r['rmse']:.2f} | {r['mape']:.2f}% | {r['wape']:.2f}% | {delta_mae:+.2f}% | {delta_rmse:+.2f}% |")

print("\n--- FIGURE 3 GINI FEATURE IMPORTANCES ---")
model = joblib.load('models/demand_forecasting_model.joblib')
prep = model.named_steps['preprocessor']
rf_est = model.named_steps['regressor']
cat_cols = prep.named_transformers_['categorical'].get_feature_names_out(['product_id', 'category']).tolist()
num_cols = [
    'inventory_lag_1', 'supplier_lead_time', 'promotion', 'price',
    'day_of_week', 'month', 'week_of_year', 'is_weekend',
    'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7'
]
raw_features = cat_cols + num_cols
importances = rf_est.feature_importances_

feature_dict = {}
for name, imp in zip(raw_features, importances):
    if name.startswith('product_id_'):
        feature_dict['product_id'] = feature_dict.get('product_id', 0.0) + imp
    elif name.startswith('category_'):
        feature_dict['category'] = feature_dict.get('category', 0.0) + imp
    else:
        feature_dict[name] = imp

print("Logical Features (14 total):")
for k, v in sorted(feature_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"| `{k}` | {v*100:.2f}% |")

groups = {
    'Rolling statistics': ['rolling_mean_7', 'rolling_std_7'],
    'Identity': ['product_id', 'category'],
    'Demand memory': ['lag_1', 'lag_7'],
    'Commercial': ['price', 'promotion'],
    'Calendar': ['day_of_week', 'month', 'week_of_year', 'is_weekend'],
    'Inventory and supply': ['inventory_lag_1', 'supplier_lead_time']
}
print("\n6 Feature Groups:")
for grp, fts in groups.items():
    tot = sum(feature_dict.get(f, 0.0) for f in fts)
    print(f"| **{grp}** | {', '.join([f'`{f}`' for f in fts])} | {tot*100:.2f}% |")

print("\n" + "="*60)
print("SECTION E: 12-PRODUCT TEST ACCURACY & INTERMITTENCY")
print("="*60)
intermittency = []
for pid in prod_metrics['product_id']:
    r_sku = raw[raw['product_id'] == pid].sort_values('date')
    t_sku = test[test['product_id'] == pid].sort_values('date')
    
    # Per-product baseline and RF RMSE
    fc_p = forecasts[forecasts['product_id'] == pid]
    yp = fc_p['actual_units_sold']
    bp = fc_p['baseline_prediction']
    rfp = fc_p['random_forest_prediction']
    b_rmse_p = np.sqrt(np.mean((yp - bp)**2))
    rf_rmse_p = np.sqrt(np.mean((yp - rfp)**2))
    nz_p = yp > 0
    b_mape_p = np.mean(np.abs((yp[nz_p] - bp[nz_p]) / yp[nz_p])) * 100 if nz_p.sum() > 0 else 0
    rf_mape_p = np.mean(np.abs((yp[nz_p] - rfp[nz_p]) / yp[nz_p])) * 100 if nz_p.sum() > 0 else 0
    
    # 365 panel zero days
    zero_365 = (r_sku['units_sold'] == 0).sum()
    zero_365_pct = zero_365 / len(r_sku) * 100
    
    # 72 test zero days
    zero_test = (t_sku['units_sold'] == 0).sum()
    zero_test_pct = zero_test / len(t_sku) * 100
    
    # ADI (Average Demand Interval = mean intervals between positive sales)
    pos_idx = np.where(r_sku['units_sold'].values > 0)[0]
    if len(pos_idx) > 1:
        intervals = np.diff(pos_idx)
        adi_mean = np.mean(intervals)
        adi_max = np.max(intervals)
    else:
        adi_mean = len(r_sku)
        adi_max = len(r_sku)
        
    intermittency.append({
        'product_id': pid,
        'b_rmse': b_rmse_p,
        'rf_rmse': rf_rmse_p,
        'b_mape': b_mape_p,
        'rf_mape': rf_mape_p,
        'zero_365': zero_365,
        'zero_365_pct': zero_365_pct,
        'zero_test': zero_test,
        'zero_test_pct': zero_test_pct,
        'adi_mean': adi_mean,
        'adi_max': adi_max
    })
df_inter = pd.DataFrame(intermittency)
merged_e = prod_metrics.merge(df_inter, on='product_id')
for idx, r in merged_e.sort_values('rf_wape').iterrows():
    print(f"| `{r['product_id']}` | {r['category']} | {r['daily_mean']:.2f} | {r['cv']*r['daily_mean']:.2f} | {r['cv']*100:.1f}% | {r['baseline_mae']:.2f} | {r['b_rmse']:.2f} | {r['b_mape']:.1f}% | {r['baseline_wape']:.2f}% | {r['rf_mae']:.2f} | {r['rf_rmse']:.2f} | {r['rf_mape']:.1f}% | {r['rf_wape']:.2f}% | {r['zero_365']} ({r['zero_365_pct']:.1f}%) | {r['zero_test']} ({r['zero_test_pct']:.1f}%) | {r['adi_mean']:.2f} (max: {r['adi_max']}) |")

print("\n" + "="*60)
print("SECTION F: INVENTORY & RECOMMENDATIONS (with Ratios & Table 6)")
print("="*60)
# Merge risks, recs, and test inventory
test_inv = test.groupby('product_id').agg(
    mean_test_inventory=('inventory_level', 'mean'),
    end_test_inventory=('inventory_level', 'last')
).reset_index()

merged_f = risks.merge(recs[['product_id', 'recommendation', 'priority']], on='product_id').merge(test_inv, on='product_id')

routing_map = {
    'Severe Overstock': 'Route 4: Capital Lockup Mitigation (Suspend PO, Launch Targeted Promo)',
    'Potential Overstock': 'Route 3: Order Truncation & Holding Review (Reduce PO Quantity)',
    'Healthy Inventory': 'Route 1: Standard Replenishment Cadence (Maintain Safety Stock)',
    'Medium Stockout Risk': 'Route 2: Buffer Augmentation (Review Safety Stock & Supplier SLA)',
    'High Stockout Risk': 'Route 5: Expedited Replenishment & Lead-Time Audit (Urgent Reorder)'
}

for idx, r in merged_f.sort_values('reorder_point', ascending=False).iterrows():
    end_ratio = r['end_test_inventory'] / r['reorder_point'] if r['reorder_point'] > 0 else 0
    mean_ratio = r['mean_test_inventory'] / r['reorder_point'] if r['reorder_point'] > 0 else 0
    route = routing_map.get(r['risk_status'], 'General Review')
    print(f"| `{r['product_id']}` | {r['category']} | {r['supplier_lead_time']}d | {r['predicted_daily_demand']:.2f} | {r['lead_time_demand']/r['supplier_lead_time']:.2f} | {r['lead_time_demand']:.2f} | {r['rolling_std_7']*np.sqrt(r['supplier_lead_time']):.2f} | {r['safety_stock']:.2f} | {r['reorder_point']:.2f} | {r['end_test_inventory']:.2f} ({end_ratio:.2f}x) | {r['mean_test_inventory']:.2f} ({mean_ratio:.2f}x) | **{r['risk_status']}** | {r['priority']} | `{r['recommendation']}` | {route} |")

print("\n" + "="*60)
print("SECTION G: PROMOTION-STRATIFIED EVALUATION")
print("="*60)
promo_mask = forecasts['promotion'] == 1
non_promo_mask = forecasts['promotion'] == 0

print(f"Test Set Total Observations: {len(forecasts)}")
print(f"Promo Days: {promo_mask.sum()} ({promo_mask.sum()/len(forecasts)*100:.1f}%), Non-Promo Days: {non_promo_mask.sum()} ({non_promo_mask.sum()/len(forecasts)*100:.1f}%)")

y_p = forecasts.loc[promo_mask, 'actual_units_sold']
b_p = forecasts.loc[promo_mask, 'baseline_prediction']
rf_p = forecasts.loc[promo_mask, 'random_forest_prediction']

y_np = forecasts.loc[non_promo_mask, 'actual_units_sold']
b_np = forecasts.loc[non_promo_mask, 'baseline_prediction']
rf_np = forecasts.loc[non_promo_mask, 'random_forest_prediction']

b_p_mae = np.mean(np.abs(y_p - b_p))
b_p_wape = np.sum(np.abs(y_p - b_p)) / np.sum(y_p) * 100
rf_p_mae = np.mean(np.abs(y_p - rf_p))
rf_p_wape = np.sum(np.abs(y_p - rf_p)) / np.sum(y_p) * 100

b_np_mae = np.mean(np.abs(y_np - b_np))
b_np_wape = np.sum(np.abs(y_np - b_np)) / np.sum(y_np) * 100
rf_np_mae = np.mean(np.abs(y_np - rf_np))
rf_np_wape = np.sum(np.abs(y_np - rf_np)) / np.sum(y_np) * 100

print(f"Panel Baseline Promo: MAE={b_p_mae:.2f}, WAPE={b_p_wape:.2f}%")
print(f"Panel Baseline Non-Promo: MAE={b_np_mae:.2f}, WAPE={b_np_wape:.2f}% (Degradation: {(b_p_mae - b_np_mae)/b_np_mae*100:+.2f}% MAE, {(b_p_wape - b_np_wape)/b_np_wape*100:+.2f}% WAPE)")
print(f"Panel RF Promo: MAE={rf_p_mae:.2f}, WAPE={rf_p_wape:.2f}%")
print(f"Panel RF Non-Promo: MAE={rf_np_mae:.2f}, WAPE={rf_np_wape:.2f}%")
print(f"Panel RF Improvement over Baseline on Promo Days: {(b_p_mae - rf_p_mae)/b_p_mae*100:.2f}% MAE, {(b_p_wape - rf_p_wape)/b_p_wape*100:.2f}% WAPE")

# Case C: HOUSEHOLD_1_418
c_fc = forecasts[forecasts['product_id'] == 'HOUSEHOLD_1_418']
c_p_mask = c_fc['promotion'] == 1
c_np_mask = c_fc['promotion'] == 0

c_y_p = c_fc.loc[c_p_mask, 'actual_units_sold']
c_b_p = c_fc.loc[c_p_mask, 'baseline_prediction']
c_rf_p = c_fc.loc[c_p_mask, 'random_forest_prediction']

c_y_np = c_fc.loc[c_np_mask, 'actual_units_sold']
c_b_np = c_fc.loc[c_np_mask, 'baseline_prediction']
c_rf_np = c_fc.loc[c_np_mask, 'random_forest_prediction']

print(f"\nCase C (HOUSEHOLD_1_418): Total={len(c_fc)}, Promo={c_p_mask.sum()}, Non-Promo={c_np_mask.sum()}")
print(f"Baseline: Promo MAE={np.mean(np.abs(c_y_p - c_b_p)):.2f}, Non-Promo MAE={np.mean(np.abs(c_y_np - c_b_np)):.2f} (Degradation: {(np.mean(np.abs(c_y_p - c_b_p)) - np.mean(np.abs(c_y_np - c_b_np)))/np.mean(np.abs(c_y_np - c_b_np))*100:+.2f}%)")
print(f"Baseline: Promo WAPE={np.sum(np.abs(c_y_p - c_b_p))/np.sum(c_y_p)*100:.2f}%, Non-Promo WAPE={np.sum(np.abs(c_y_np - c_b_np))/np.sum(c_y_np)*100:.2f}%")
print(f"RF:       Promo MAE={np.mean(np.abs(c_y_p - c_rf_p)):.2f}, Non-Promo MAE={np.mean(np.abs(c_y_np - c_rf_np)):.2f}")
print(f"RF:       Promo WAPE={np.sum(np.abs(c_y_p - c_rf_p))/np.sum(c_y_p)*100:.2f}%, Non-Promo WAPE={np.sum(np.abs(c_y_np - c_rf_np))/np.sum(c_y_np)*100:.2f}%")

print("\nAll 12 SKUs Promo vs Non-Promo:")
for pid in sorted(forecasts['product_id'].unique()):
    p_df = forecasts[forecasts['product_id'] == pid]
    pm = p_df['promotion'] == 1
    npm = p_df['promotion'] == 0
    
    yp = p_df.loc[pm, 'actual_units_sold']
    bp = p_df.loc[pm, 'baseline_prediction']
    rfp = p_df.loc[pm, 'random_forest_prediction']
    
    ynp = p_df.loc[npm, 'actual_units_sold']
    bnp = p_df.loc[npm, 'baseline_prediction']
    rfnp = p_df.loc[npm, 'random_forest_prediction']
    
    bp_mae = np.mean(np.abs(yp - bp)) if len(yp) > 0 else 0
    bp_wape = np.sum(np.abs(yp - bp)) / np.sum(yp) * 100 if np.sum(yp) > 0 else 0
    rfp_mae = np.mean(np.abs(yp - rfp)) if len(yp) > 0 else 0
    rfp_wape = np.sum(np.abs(yp - rfp)) / np.sum(yp) * 100 if np.sum(yp) > 0 else 0
    
    bnp_mae = np.mean(np.abs(ynp - bnp)) if len(ynp) > 0 else 0
    bnp_wape = np.sum(np.abs(ynp - bnp)) / np.sum(ynp) * 100 if np.sum(ynp) > 0 else 0
    rfnp_mae = np.mean(np.abs(ynp - rfnp)) if len(ynp) > 0 else 0
    rfnp_wape = np.sum(np.abs(ynp - rfnp)) / np.sum(ynp) * 100 if np.sum(ynp) > 0 else 0
    
    print(f"| `{pid}` | {p_df['category'].iloc[0]} | {pm.sum()}d | {bp_mae:.2f} | {bp_wape:.1f}% | {bnp_mae:.2f} | {bnp_wape:.1f}% | {rfp_mae:.2f} | {rfp_wape:.1f}% | {rfnp_mae:.2f} | {rfnp_wape:.1f}% |")
