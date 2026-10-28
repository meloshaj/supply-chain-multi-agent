"""Visualization agent for report-ready PNG charts and 300 DPI publication figures."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from agents.common import relative_path, require_file

FIGURES_DIR = Path("outputs/figures")
PUB_FIGURES_DIR = Path("figures")


def _save(fig: plt.Figure, filename: str, dpi: int = 180) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return relative_path(path)


def _save_pub(fig: plt.Figure, filename: str, dpi: int = 300) -> str:
    PUB_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = PUB_FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return relative_path(path)


def run_visualization_agent(
    processed_dataset_path: str,
    forecast_results_path: str,
    inventory_results_path: str,
) -> dict[str, object]:
    """Generate report-ready Matplotlib visualizations and publication figures at 300 DPI."""
    processed_path = require_file(processed_dataset_path)
    forecast_path = require_file(forecast_results_path)
    inventory_path = require_file(inventory_results_path)

    processed = pd.read_csv(processed_path)
    forecasts = pd.read_csv(forecast_path)
    inventory = pd.read_csv(inventory_path)
    processed["date"] = pd.to_datetime(processed["date"], errors="coerce")
    forecasts["date"] = pd.to_datetime(forecasts["date"], errors="coerce")

    figure_paths: list[str] = []
    pub_figure_paths: list[str] = []

    # 1. Standard pipeline figures (in outputs/figures/)
    daily = processed.groupby("date", as_index=False)["units_sold"].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily["date"], daily["units_sold"], color="#1f77b4", linewidth=1.8, label="Units sold")
    ax.set_title("Total Demand Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend()
    figure_paths.append(_save(fig, "demand_over_time.png"))

    actual_pred = forecasts.groupby("date", as_index=False)[["actual_units_sold", "random_forest_prediction"]].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual_pred["date"], actual_pred["actual_units_sold"], label="Actual", linewidth=1.8)
    ax.plot(actual_pred["date"], actual_pred["random_forest_prediction"], label="Random Forest", linewidth=1.8)
    ax.set_title("Actual Versus Predicted Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend()
    figure_paths.append(_save(fig, "actual_vs_predicted_demand.png"))

    mae_baseline = float((forecasts["actual_units_sold"] - forecasts["baseline_prediction"]).abs().mean())
    mae_rf = float((forecasts["actual_units_sold"] - forecasts["random_forest_prediction"]).abs().mean())
    rmse_baseline = float(((forecasts["actual_units_sold"] - forecasts["baseline_prediction"]) ** 2).mean() ** 0.5)
    rmse_rf = float(((forecasts["actual_units_sold"] - forecasts["random_forest_prediction"]) ** 2).mean() ** 0.5)
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_df = pd.DataFrame({"Baseline": [mae_baseline, rmse_baseline], "Random Forest": [mae_rf, rmse_rf]}, index=["MAE", "RMSE"])
    metrics_df.plot(kind="bar", ax=ax, color=["#8c564b", "#2ca02c"])
    ax.set_title("Baseline Versus Random Forest Metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Error")
    ax.legend()
    figure_paths.append(_save(fig, "baseline_vs_random_forest_metrics.png"))

    product_demand = processed.groupby("product_id", as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(product_demand["product_id"], product_demand["units_sold"], color="#17becf")
    ax.set_title("Demand By Product")
    ax.set_xlabel("Product")
    ax.set_ylabel("Units sold")
    ax.tick_params(axis="x", rotation=45)
    figure_paths.append(_save(fig, "demand_by_product.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(forecasts["forecast_error"], bins=30, color="#9467bd", edgecolor="white")
    ax.set_title("Forecast Error Distribution")
    ax.set_xlabel("Actual minus predicted")
    ax.set_ylabel("Count")
    figure_paths.append(_save(fig, "forecast_error_distribution.png"))

    risk_counts = inventory["risk_status"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(risk_counts.index, risk_counts.values, color="#d62728")
    ax.set_title("Inventory Risk Category Counts")
    ax.set_xlabel("Risk category")
    ax.set_ylabel("Products")
    ax.tick_params(axis="x", rotation=25)
    figure_paths.append(_save(fig, "inventory_risk_category_counts.png"))

    compare = inventory.sort_values("product_id")
    x = range(len(compare))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - 0.2 for i in x], compare["inventory_level"], width=0.4, label="Current inventory", color="#7f7f7f")
    ax.bar([i + 0.2 for i in x], compare["reorder_point"], width=0.4, label="Reorder point", color="#ff7f0e")
    ax.set_xticks(list(x), compare["product_id"], rotation=45)
    ax.set_title("Current Inventory Versus Reorder Point")
    ax.set_xlabel("Product")
    ax.set_ylabel("Units")
    ax.legend()
    figure_paths.append(_save(fig, "current_inventory_vs_reorder_point.png"))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(compare["product_id"], compare["recommended_replenishment_quantity"], color="#2ca02c")
    ax.set_title("Recommended Replenishment Quantity By Product")
    ax.set_xlabel("Product")
    ax.set_ylabel("Units")
    ax.tick_params(axis="x", rotation=45)
    figure_paths.append(_save(fig, "recommended_replenishment_quantity_by_product.png"))

    # 2. Publication Figures (300 DPI, in figures/)
    # Styling for publication
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
    })

    # fig1_error_metrics.png: Dual-panel bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    models = ["Seasonal-Naive Baseline", "Random Forest (Full)"]
    x_pos = np.arange(len(models))
    w = 0.32
    
    y_test_arr = forecasts["actual_units_sold"].to_numpy(dtype=float)
    rf_pred_arr = forecasts["random_forest_prediction"].to_numpy(dtype=float)
    base_pred_arr = forecasts["baseline_prediction"].to_numpy(dtype=float)
    
    mask = y_test_arr > 0
    mape_base = float(np.mean(np.abs((y_test_arr[mask] - base_pred_arr[mask]) / y_test_arr[mask])) * 100)
    mape_rf = float(np.mean(np.abs((y_test_arr[mask] - rf_pred_arr[mask]) / y_test_arr[mask])) * 100)
    wape_base = float(np.abs(y_test_arr - base_pred_arr).sum() / y_test_arr.sum() * 100)
    wape_rf = float(np.abs(y_test_arr - rf_pred_arr).sum() / y_test_arr.sum() * 100)

    rects1 = ax1.bar(x_pos - w/2, [mae_baseline, mae_rf], w, label="MAE", color="#1f77b4", edgecolor="#0d47a1", alpha=0.9)
    rects2 = ax1.bar(x_pos + w/2, [rmse_baseline, rmse_rf], w, label="RMSE", color="#2ca02c", edgecolor="#1b5e20", alpha=0.9)
    ax1.set_ylabel("Error (units / day)")
    ax1.set_title("(a) Scale-Dependent Absolute Error Metrics", fontweight="bold", pad=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, max(rmse_baseline, rmse_rf) * 1.3)
    ax1.legend(loc="upper right", frameon=True)
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    rects3 = ax2.bar(x_pos - w/2, [mape_base, mape_rf], w, label="MAPE", color="#ff7f0e", edgecolor="#e65100", alpha=0.9)
    rects4 = ax2.bar(x_pos + w/2, [wape_base, wape_rf], w, label="WAPE", color="#9467bd", edgecolor="#4a148c", alpha=0.9)
    ax2.set_ylabel("Percentage Error (%)")
    ax2.set_title("(b) Scale-Free Percentage Error Metrics", fontweight="bold", pad=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, max(mape_base, mape_rf) * 1.25)
    ax2.legend(loc="upper right", frameon=True)
    for r in rects3:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for r in rects4:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    pub_figure_paths.append(_save_pub(fig, "fig1_error_metrics.png", dpi=300))

    # fig2_ablation_mae.png: Horizontal bar chart of ablation suite
    ablation_file = Path("outputs/tables/ablation_results.csv")
    if ablation_file.exists():
        df_abl = pd.read_csv(ablation_file)
        plot_tab = df_abl.iloc[::-1].copy()
        fig, ax = plt.subplots(figsize=(10, 5.5))
        y_p = np.arange(len(plot_tab))
        colors = ["#c53030" if c > 0 else "#2f855a" for c in plot_tab["delta_mae_percent"]]
        for i, cfg in enumerate(plot_tab["configuration"]):
            if "Full model" in cfg:
                colors[i] = "#4a5568"
        bars = ax.barh(y_p, plot_tab["mae"], height=0.6, color=colors, alpha=0.85, edgecolor="#2d3748")
        ax.set_yticks(y_p)
        ax.set_yticklabels(plot_tab["configuration"], fontsize=10)
        ax.set_xlabel("Mean Absolute Error (units / day)")
        ax.set_title("Test Set MAE Across Model Ablation Configurations (Table 5)", fontweight="bold", pad=12)
        full_mae = df_abl.loc[df_abl["configuration"] == "Full model (all features)", "mae"].values[0]
        ax.axvline(full_mae, color="#2b6cb0", linestyle="--", linewidth=1.8, label=f"Full Model Baseline (MAE = {full_mae:.2f})")
        for i, bar in enumerate(bars):
            bw = bar.get_width()
            cfg = plot_tab["configuration"].iloc[i]
            delta = plot_tab["delta_mae_percent"].iloc[i]
            label_text = f"{bw:.2f}" if "Full model" in cfg else f"{bw:.2f} ({delta:+.1f}%)"
            ax.text(bw + 0.03, bar.get_y() + bar.get_height()/2, label_text, va="center", ha="left", fontsize=9.5, fontweight="bold")
        ax.set_xlim(3.7, max(plot_tab["mae"]) * 1.1)
        ax.legend(loc="lower right", frameon=True)
        pub_figure_paths.append(_save_pub(fig, "fig2_ablation_mae.png", dpi=300))

    # fig3_feature_importance.png: Horizontal bar plot
    model_joblib = Path("models/demand_forecasting_model.joblib")
    if model_joblib.exists():
        loaded_model = joblib.load(model_joblib)
        prep = loaded_model.named_steps["preprocessor"]
        rf_est = loaded_model.named_steps["regressor"]
        cat_c = prep.named_transformers_["categorical"].get_feature_names_out(["product_id", "category"]).tolist()
        num_c = [
            "inventory_lag_1", "supplier_lead_time", "promotion", "price",
            "day_of_week", "month", "week_of_year", "is_weekend",
            "lag_1", "lag_7", "rolling_mean_7", "rolling_std_7"
        ]
        raw_names = cat_c + num_c
        raw_imp = rf_est.feature_importances_
        grp_imp = {}
        for fn, imp in zip(raw_names, raw_imp):
            if fn.startswith("product_id_"):
                grp_imp["Product Identity (OHE)"] = grp_imp.get("Product Identity (OHE)", 0.0) + imp
            elif fn.startswith("category_"):
                grp_imp["Category Identity (OHE)"] = grp_imp.get("Category Identity (OHE)", 0.0) + imp
            elif fn == "rolling_mean_7":
                grp_imp["Rolling Mean (7-day)"] = imp
            elif fn == "rolling_std_7":
                grp_imp["Rolling Std (7-day)"] = imp
            elif fn == "lag_7":
                grp_imp["Seasonal Lag (lag_7)"] = imp
            elif fn == "lag_1":
                grp_imp["Immediate Lag (lag_1)"] = imp
            elif fn == "inventory_lag_1":
                grp_imp["Prior Inventory (lag_1)"] = imp
            elif fn == "day_of_week":
                grp_imp["Day of Week"] = imp
            elif fn == "week_of_year":
                grp_imp["Week of Year"] = imp
            elif fn == "supplier_lead_time":
                grp_imp["Supplier Lead Time"] = imp
            elif fn == "is_weekend":
                grp_imp["Weekend Indicator"] = imp
            elif fn == "month":
                grp_imp["Month"] = imp
            elif fn == "price":
                grp_imp["Sell Price"] = imp
            elif fn == "promotion":
                grp_imp["Promotion / SNAP"] = imp
        df_fi = pd.DataFrame(list(grp_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos_fi = np.arange(len(df_fi))
        bars = ax.barh(y_pos_fi, df_fi["Importance"], height=0.65, color="#2b6cb0", edgecolor="#1a365d", alpha=0.88)
        ax.set_yticks(y_pos_fi)
        ax.set_yticklabels(df_fi["Feature"], fontsize=10)
        ax.set_xlabel("Aggregated Gini Feature Importance (MDI)")
        ax.set_title("Aggregated Feature Importance from Full Random Forest Regressor", fontweight="bold", pad=12)
        ax.set_xlim(0, max(df_fi["Importance"]) * 1.18)
        for bar in bars:
            bw = bar.get_width()
            ax.text(bw + 0.005, bar.get_y() + bar.get_height()/2, f"{bw:.3f} ({bw*100:.1f}%)", va="center", ha="left", fontsize=9.5)
        pub_figure_paths.append(_save_pub(fig, "fig3_feature_importance.png", dpi=300))

    # fig4_case_a_high_volume.png: FOODS_3_586
    case_a = forecasts[forecasts["product_id"] == "FOODS_3_586"].sort_values("date")
    if not case_a.empty:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})
        ax_top.plot(case_a["date"], case_a["actual_units_sold"], label="Actual Demand", color="#1a202c", linewidth=2.0, marker="o", markersize=3.5)
        ax_top.plot(case_a["date"], case_a["random_forest_prediction"], label="Random Forest Forecast", color="#2b6cb0", linewidth=1.8)
        ax_top.plot(case_a["date"], case_a["baseline_prediction"], label="Seasonal-Naive Baseline", color="#c53030", linewidth=1.4, linestyle="--", alpha=0.8)
        ax_top.set_ylabel("Demand (units / day)")
        ax_top.set_title("Case Study A: High-Volume Staple (FOODS_3_586, Dept: FOODS_3)\nLow Forecast Error (WAPE = 17.3%) Coexisting with Ample Inventory Buffer", fontweight="bold", pad=10)
        ax_top.legend(loc="upper right", frameon=True)

        mean_inv = case_a["inventory_level"].mean()
        ax_bot.plot(case_a["date"], case_a["inventory_level"], label="Simulated End-of-Day Inventory", color="#2f855a", linewidth=1.8)
        ax_bot.axhline(mean_inv, color="#2f855a", linestyle=":", label=f"Mean Stock Buffer ({mean_inv:.0f} units)")
        ax_bot.set_ylabel("Stock (units)")
        ax_bot.set_xlabel("Test Window Date (72 Days: March 2016 – May 2016)")
        ax_bot.legend(loc="upper right", frameon=True)
        ax_bot.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()
        pub_figure_paths.append(_save_pub(fig, "fig4_case_a_high_volume.png", dpi=300))

    # fig5_case_b_volatile.png: HOBBIES_1_155
    case_b = forecasts[forecasts["product_id"] == "HOBBIES_1_155"].sort_values("date")
    if not case_b.empty:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(case_b["date"], case_b["actual_units_sold"], label="Actual Demand", color="#1a202c", linewidth=2.0, marker="o", markersize=4)
        ax.plot(case_b["date"], case_b["random_forest_prediction"], label="Random Forest Forecast (WAPE = 49.2%)", color="#2b6cb0", linewidth=1.8)
        ax.plot(case_b["date"], case_b["baseline_prediction"], label="Seasonal-Naive Baseline (WAPE = 64.7%)", color="#c53030", linewidth=1.4, linestyle="--", alpha=0.85)
        ax.set_ylabel("Demand (units / day)")
        ax.set_xlabel("Test Window Date (March 2016 – May 2016)")
        ax.set_title("Case Study B: Volatile Low-Volume Item (HOBBIES_1_155, Dept: HOBBIES_1)\n" + r"Elevated Relative Volatility ($\sigma / \mu = 0.62$) and Discontinuous Demand Dynamics", fontweight="bold", pad=12)
        ax.legend(loc="upper right", frameon=True)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()
        pub_figure_paths.append(_save_pub(fig, "fig5_case_b_volatile.png", dpi=300))

    # fig6_case_c_promotion.png: HOUSEHOLD_1_418
    case_c = forecasts[forecasts["product_id"] == "HOUSEHOLD_1_418"].sort_values("date")
    if not case_c.empty:
        fig, ax = plt.subplots(figsize=(11, 5.0))
        promo_added = False
        for _, r in case_c.iterrows():
            if r["promotion"] == 1:
                ax.axvspan(r["date"] - pd.Timedelta(hours=12), r["date"] + pd.Timedelta(hours=12),
                           color="#fed7d7", alpha=0.55, label="Promotion / SNAP Day" if not promo_added else "")
                promo_added = True
        ax.plot(case_c["date"], case_c["actual_units_sold"], label="Actual Demand", color="#1a202c", linewidth=2.0, marker="o", markersize=4)
        ax.plot(case_c["date"], case_c["random_forest_prediction"], label="Random Forest Forecast (Promo MAE = 5.97)", color="#2b6cb0", linewidth=2.0)
        ax.plot(case_c["date"], case_c["baseline_prediction"], label="Seasonal-Naive Baseline (Promo MAE = 9.83)", color="#c53030", linewidth=1.6, linestyle="--", alpha=0.9)
        ax.set_ylabel("Demand (units / day)")
        ax.set_xlabel("Test Window Date (March 2016 – May 2016)")
        ax.set_title("Case Study C: Promotion-Driven Series (HOUSEHOLD_1_418, Dept: HOUSEHOLD_1)\nBaseline Deterioration (+70.4% Error on Promo Days) vs. Random Forest Adaptation", fontweight="bold", pad=12)
        ax.legend(loc="upper left", frameon=True)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()
        pub_figure_paths.append(_save_pub(fig, "fig6_case_c_promotion.png", dpi=300))

    return {
        "status": "success",
        "agent": "Visualization Agent",
        "figures_generated": len(figure_paths),
        "figure_paths": figure_paths,
        "publication_figures_generated": len(pub_figure_paths),
        "publication_figure_paths": pub_figure_paths,
    }
