#!/usr/bin/env python3
"""
Analyze simulation results from performSimulations.py

This script provides comprehensive analysis and visualization of simulation
statistics including:
- Overall effectiveness of recommendations
- Action pair analysis
- Strategy comparison breakdowns
- Liquidation reason analysis
- Per-strategy performance metrics
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, visualizations will be skipped")


def load_statistics(json_file):
    """Load statistics from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def print_summary_table(stats, title="Summary"):
    """Print a formatted summary table."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    
    overall = stats.get('overall', {})
    processed = overall.get('processed', 0)
    
    if processed == 0:
        print("No data to analyze")
        return
    
    print(f"\n{'Metric':<50} {'Value':<15} {'Percentage':<15}")
    print("-" * 80)
    
    metrics = [
        ("Total Processed", "processed"),
        ("Liquidated WITHOUT recommendation", "liquidated_without"),
        ("Liquidated WITH recommendation", "liquidated_with"),
        ("Improved (avoided liquidation)", "improved"),
        ("Worsened (introduced liquidation)", "worsened"),
        ("No change", "no_change"),
        ("  - Both liquidated", "no_change_with_liquidation"),
        ("  - Neither liquidated", "no_change_without_liquidation"),
    ]
    
    for label, key in metrics:
        value = overall.get(key, 0)
        pct = (value / processed * 100) if processed > 0 else 0
        print(f"{label:<50} {value:<15} {pct:>6.2f}%")
    
    # Effectiveness metrics
    improved = overall.get('improved', 0)
    worsened = overall.get('worsened', 0)
    net_benefit = improved - worsened
    
    print("-" * 80)
    print(f"{'Net Benefit (Improved - Worsened)':<50} {net_benefit:<15}")
    
    liq_without = overall.get('liquidated_without', 0)
    liq_with = overall.get('liquidated_with', 0)
    
    if liq_without > 0:
        reduction_pct = (improved / liq_without * 100) if liq_without > 0 else 0
        print(f"{'Liquidation Reduction Rate':<50} {reduction_pct:>15.2f}%")
    
    # Risk prediction accuracy
    at_risk = overall.get('at_risk', 0)
    not_at_risk = overall.get('not_at_risk', 0)
    not_at_risk_but_liquidated = overall.get('not_at_risk_but_liquidated', 0)
    
    if at_risk + not_at_risk > 0:
        print(f"\n{'Risk Prediction':<50}")
        print(f"{'  Total At Risk':<50} {at_risk:<15}")
        print(f"{'  Total Not At Risk':<50} {not_at_risk:<15}")
        if not_at_risk > 0:
            false_negative_rate = (not_at_risk_but_liquidated / not_at_risk * 100) if not_at_risk > 0 else 0
            print(f"{'  False Negatives (not at risk but liquidated)':<50} {not_at_risk_but_liquidated:<15} ({false_negative_rate:.2f}%)")


def analyze_liquidation_reasons(stats):
    """Analyze liquidation reasons breakdown."""
    overall = stats.get('overall', {})
    
    print(f"\n{'='*80}")
    print("Liquidation Reason Breakdown".center(80))
    print(f"{'='*80}")
    
    liq_without = overall.get('liquidated_without', 0)
    liq_with = overall.get('liquidated_with', 0)
    
    # Dust liquidations
    dust_without = overall.get('dust_liquidations_without', 0)
    dust_with = overall.get('dust_liquidations_with', 0)
    
    # HF-based liquidations
    hf_without = overall.get('hf_based_liquidations_without', 0)
    hf_with = overall.get('hf_based_liquidations_with', 0)
    
    # Threshold-based liquidations
    threshold_without = overall.get('threshold_based_liquidations_without', 0)
    threshold_with = overall.get('threshold_based_liquidations_with', 0)
    
    print(f"\n{'Type':<30} {'Without Rec':<20} {'With Rec':<20} {'Change':<10}")
    print("-" * 80)
    
    if liq_without > 0:
        print(f"{'Total Liquidations':<30} {liq_without:<20} {liq_with:<20} {liq_with - liq_without:+d}")
        print(f"{'  Dust':<30} {dust_without:<20} ({dust_without/liq_without*100:.1f}%) {dust_with:<20} ({dust_with/liq_with*100:.1f}% if >0) {dust_with - dust_without:+d}")
        print(f"{'  HF-based':<30} {hf_without:<20} ({hf_without/liq_without*100:.1f}%) {hf_with:<20} ({hf_with/liq_with*100:.1f}% if >0) {hf_with - hf_without:+d}")
        print(f"{'  Threshold-based':<30} {threshold_without:<20} ({threshold_without/liq_without*100:.1f}%) {threshold_with:<20} ({threshold_with/liq_with*100:.1f}% if >0) {threshold_with - threshold_without:+d}")
    
    # Sample reasons
    reasons_without = overall.get('liquidation_reasons_without', [])
    reasons_with = overall.get('liquidation_reasons_with', [])
    
    if reasons_without:
        print(f"\nSample Liquidation Reasons (WITHOUT recommendation):")
        unique_reasons = Counter(reasons_without).most_common(5)
        for reason, count in unique_reasons:
            print(f"  - {reason}: {count} times")


def analyze_action_pairs(stats):
    """Analyze statistics by action pair."""
    by_pair = stats.get('by_action_pair', {})
    
    if not by_pair:
        return
    
    print(f"\n{'='*80}")
    print("Action Pair Analysis".center(80))
    print(f"{'='*80}")
    
    # Calculate key metrics for each pair
    pair_metrics = []
    
    for pair_key, pair_stats in by_pair.items():
        processed = pair_stats.get('processed', 0)
        if processed == 0:
            continue
        
        improved = pair_stats.get('improved', 0)
        worsened = pair_stats.get('worsened', 0)
        liq_without = pair_stats.get('liquidated_without', 0)
        liq_with = pair_stats.get('liquidated_with', 0)
        
        improvement_rate = (improved / processed * 100) if processed > 0 else 0
        net_benefit = improved - worsened
        
        pair_metrics.append({
            'pair': pair_key,
            'processed': processed,
            'improved': improved,
            'worsened': worsened,
            'net_benefit': net_benefit,
            'improvement_rate': improvement_rate,
            'liq_without': liq_without,
            'liq_with': liq_with,
        })
    
    # Sort by net benefit
    pair_metrics.sort(key=lambda x: x['net_benefit'], reverse=True)
    
    print(f"\n{'Action Pair':<30} {'Processed':<12} {'Improved':<12} {'Worsened':<12} {'Net':<12} {'Rate':<12}")
    print("-" * 90)
    
    for m in pair_metrics[:15]:  # Top 15
        print(f"{m['pair']:<30} {m['processed']:<12} {m['improved']:<12} {m['worsened']:<12} "
              f"{m['net_benefit']:+12} {m['improvement_rate']:>6.2f}%")


def analyze_strategy_comparison(stats):
    """Analyze multi-strategy liquidation detection results."""
    overall = stats.get('overall', {})
    strategy_comps = overall.get('strategy_comparisons', {})
    
    if not strategy_comps:
        return
    
    print(f"\n{'='*80}")
    print("Strategy Comparison Analysis".center(80))
    print(f"{'='*80}")
    
    # Consensus agreement
    consensus_without = strategy_comps.get('consensus_agreement_without', [])
    consensus_with = strategy_comps.get('consensus_agreement_with', [])
    
    if consensus_without:
        avg_agreement_without = np.mean(consensus_without) * 100
        print(f"\nAverage Consensus Agreement (WITHOUT recommendation): {avg_agreement_without:.1f}%")
    
    if consensus_with:
        avg_agreement_with = np.mean(consensus_with) * 100
        print(f"Average Consensus Agreement (WITH recommendation): {avg_agreement_with:.1f}%")
    
    # Strategy performance
    for scenario in ['without', 'with']:
        strategy_stats = strategy_comps.get(scenario, {})
        if not strategy_stats:
            continue
        
        processed = overall.get('processed', 0)
        if processed == 0:
            continue
        
        print(f"\nStrategy Performance ({scenario.upper()} recommendation):")
        print(f"{'Strategy':<25} {'Detected':<12} {'Rate':<12} {'Avg Checks':<15} {'Avg Time (days)':<15}")
        print("-" * 80)
        
        for strategy_name, s_stats in sorted(strategy_stats.items()):
            detected = s_stats.get('detected', 0)
            checks = s_stats.get('checks', 0)
            times = s_stats.get('time', [])
            
            detection_rate = (detected / processed * 100) if processed > 0 else 0
            avg_checks = checks / processed if processed > 0 else 0
            avg_time = np.mean(times) / (24 * 3600) if times else None
            
            time_str = f"{avg_time:.2f}" if avg_time is not None else "N/A"
            print(f"{strategy_name:<25} {detected:<12} {detection_rate:>6.2f}% "
                  f"{avg_checks:>8.1f} {time_str:>15}")
    
    # Best strategy counts
    best_strategy_counts = strategy_comps.get('best_strategy_counts', {})
    if best_strategy_counts:
        for scenario in ['without', 'with']:
            counts = best_strategy_counts.get(scenario, {})
            if counts:
                print(f"\nMost Frequently Best Strategy ({scenario.upper()} recommendation):")
                sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                for strategy, count in sorted_counts[:5]:
                    pct = (count / processed * 100) if processed > 0 else 0
                    print(f"  - {strategy}: {count} times ({pct:.1f}%)")


def analyze_time_deltas(stats):
    """Analyze time-to-liquidation deltas."""
    overall = stats.get('overall', {})
    deltas = overall.get("time_deltas", [])
    
    if not deltas:
        print("\nNo time-to-liquidation delta data available.")
        return

    print(f"\n{'='*80}")
    print("Time-to-Liquidation Deltas (With - Without)".center(80))
    print(f"{'='*80}")

    try:
        deltas_array = np.array(deltas, dtype=float)
        avg = float(np.mean(deltas_array))
        sd = float(np.std(deltas_array))
        median = float(np.median(deltas_array))
        
        print(f"Count: {len(deltas)}")
        print(f"Average Delta: {avg:.2f}s")
        print(f"Median Delta:  {median:.2f}s")
        print(f"Std Dev:       {sd:.2f}s")
        print(f"Min:           {min(deltas):.2f}s")
        print(f"Max:           {max(deltas):.2f}s")

        # Breakdown by direction
        positive = deltas_array[deltas_array > 0]
        negative = deltas_array[deltas_array < 0]
        
        print(f"\nBreakdown:")
        print(f"  Delayed Liquidation (> 0s):     {len(positive)} ({len(positive)/len(deltas):.1%})")
        if len(positive) > 0:
            print(f"    Avg Delay: {np.mean(positive):.2f}s ({np.mean(positive)/3600:.2f}h)")
            print(f"    Max Delay: {np.max(positive):.2f}s ({np.max(positive)/3600:.2f}h)")
            
        print(f"  Accelerated Liquidation (< 0s): {len(negative)} ({len(negative)/len(deltas):.1%})")
        if len(negative) > 0:
            print(f"    Avg Acceleration: {abs(np.mean(negative)):.2f}s ({abs(np.mean(negative))/3600:.2f}h)")
            print(f"    Max Acceleration: {abs(np.min(negative)):.2f}s ({abs(np.min(negative))/3600:.2f}h)")
    except Exception as e:
        print(f"Could not compute time-delta stats: {e}")


def analyze_risk_predictions(stats):
    """Analyze risk prediction accuracy."""
    overall = stats.get('overall', {})
    
    print(f"\n{'='*80}")
    print("Liquidation Risk Prediction Analysis".center(80))
    print(f"{'='*80}")
    
    at_risk = overall.get("at_risk", 0)
    not_at_risk = overall.get("not_at_risk", 0)
    total_risk_assessed = at_risk + not_at_risk
    
    if total_risk_assessed == 0:
        print("No risk assessment data available.")
        return

    print(f"Total recommendations assessed for risk: {total_risk_assessed}")
    
    # --- Eventual Liquidation Metrics ---
    print(f"\n1. Eventual Liquidation Prediction (Time Horizon: Any future time)")
    print("-" * 60)
    
    tp_eventual = overall.get("at_risk_eventual_liquidation_count", 0)
    fp_eventual = overall.get("at_risk_no_future_liquidation_count", 0)
    fn_eventual = overall.get("not_at_risk_but_liquidated", 0)
    # TN is approximate: not_at_risk includes those with no future, assuming they are safe
    tn_eventual = not_at_risk - fn_eventual
    
    precision_eventual = tp_eventual / (tp_eventual + fp_eventual) if (tp_eventual + fp_eventual) > 0 else 0.0
    recall_eventual = tp_eventual / (tp_eventual + fn_eventual) if (tp_eventual + fn_eventual) > 0 else 0.0
    f1_eventual = 2 * (precision_eventual * recall_eventual) / (precision_eventual + recall_eventual) if (precision_eventual + recall_eventual) > 0 else 0.0
    accuracy_eventual = (tp_eventual + tn_eventual) / total_risk_assessed if total_risk_assessed > 0 else 0.0

    print(f"  Confusion Matrix:")
    print(f"    TP (Predicted Risk -> Liquidated):      {tp_eventual}")
    print(f"    FP (Predicted Risk -> Not Liquidated):  {fp_eventual}")
    print(f"    FN (Predicted Safe -> Liquidated):      {fn_eventual}")
    print(f"    TN (Predicted Safe -> Not Liquidated):  {tn_eventual}")
    
    print(f"\n  Metrics:")
    print(f"    Precision: {precision_eventual:.2%} (Reliability of risk flag)")
    print(f"    Recall:    {recall_eventual:.2%} (Coverage of actual liquidations)")
    print(f"    F1 Score:  {f1_eventual:.4f}")
    print(f"    Accuracy:  {accuracy_eventual:.2%}")

    # --- Immediate Liquidation Metrics ---
    print(f"\n2. Immediate Liquidation Prediction (Time Horizon: Next transaction)")
    print("-" * 60)
    
    at_immediate_risk = overall.get("at_immediate_risk", 0)
    
    if at_immediate_risk > 0:
        tp_immediate = overall.get("immediate_risk_followed_by_liquidation", 0)
        fp_immediate = overall.get("immediate_risk_not_followed_by_liquidation", 0)
        # We don't track FN for immediate risk explicitly in current stats
        
        precision_immediate = tp_immediate / (tp_immediate + fp_immediate) if (tp_immediate + fp_immediate) > 0 else 0.0
        
        print(f"  Confusion Matrix (Partial):")
        print(f"    TP (Predicted Immediate -> Next is Liq):     {tp_immediate}")
        print(f"    FP (Predicted Immediate -> Next is Not Liq): {fp_immediate}")
        print(f"    FN (Predicted Safe -> Next is Liq):          N/A (Not tracked)")
        
        print(f"\n  Metrics:")
        print(f"    Precision: {precision_immediate:.2%} (Reliability of immediate risk flag)")
        print(f"    Recall:    N/A")
    else:
        print("  No immediate risk predictions made.")

    # --- Time to Liquidation Stats ---
    times_to_liq = overall.get("at_risk_time_to_liquidation", [])
    times_missed = overall.get("not_at_risk_time_to_liquidation", [])
    
    if times_to_liq or times_missed:
        print(f"\n3. Time to Liquidation Analysis (Actual time until liquidation occurred)")
        print("-" * 60)
        
        def print_time_stats(times, label):
            if not times:
                print(f"  {label}: No data")
                return
            arr = np.array(times)
            avg_days = np.mean(arr) / (24 * 3600)
            med_days = np.median(arr) / (24 * 3600)
            std_days = np.std(arr) / (24 * 3600)
            min_days = np.min(arr) / (24 * 3600)
            max_days = np.max(arr) / (24 * 3600)
            print(f"  {label} (n={len(times)}):")
            print(f"    Mean:   {avg_days:.2f} days")
            print(f"    Median: {med_days:.2f} days")
            print(f"    StdDev: {std_days:.2f} days")
            print(f"    Range:  {min_days:.2f} - {max_days:.2f} days")
            
        print_time_stats(times_to_liq, "Correctly Predicted (True Positives)")
        print_time_stats(times_missed, "Missed Predictions (False Negatives)")
        
        if times_to_liq and times_missed:
            m1 = np.mean(times_to_liq)
            m0 = np.mean(times_missed)
            diff_days = (m1 - m0) / (24 * 3600)
            
            print(f"\n  Comparison:")
            direction = "LATER" if diff_days > 0 else "SOONER"
            print(f"    Difference in means (TP - FN): {diff_days:.2f} days")
            print(f"    Interpretation: Correctly predicted liquidations happen {abs(diff_days):.2f} days {direction} than missed ones.")

    # --- Next Action Prediction ---
    print(f"\n4. Next Action Prediction Accuracy")
    print("-" * 60)
    total_mrp = overall.get("total_predictions_checked_mrp", 0)
    matches_mrp = overall.get("prediction_matches_next_action_mrp", 0)
    if total_mrp > 0:
        print(f"  Based on `most_recent_predictions` (lowest time-to-event):")
        print(f"    Correctly predicted next action: {matches_mrp} / {total_mrp} ({matches_mrp/total_mrp:.1%})")

    total_ts = overall.get("total_predictions_checked_ts", 0)
    matches_ts = overall.get("prediction_matches_next_action_ts", 0)
    if total_ts > 0:
        print(f"  Based on `trend_slopes` (most negative slope):")
        print(f"    Correctly predicted next action: {matches_ts} / {total_ts} ({matches_ts/total_ts:.1%})")


def analyze_per_simulation_breakdown(stats):
    """Analyze per-simulation strategy breakdowns."""
    overall = stats.get('overall', {})
    strategy_comps = overall.get('strategy_comparisons', {})
    
    per_sim_without = strategy_comps.get("per_simulation_without", [])
    per_sim_with = strategy_comps.get("per_simulation_with", [])

    if not (per_sim_without or per_sim_with):
        return

    print(f"\n{'='*80}")
    print("Per-Simulation Strategy Breakdown".center(80))
    print(f"{'='*80}")

    def print_samples(samples, title):
        print(f"\n{title}")
        if not samples:
            print("  No cases found.")
            return
            
        for i, sim in enumerate(samples[:10]):
            user_id = sim.get("user", "unknown")
            detected = sim.get("strategies_detected", [])
            not_detected = sim.get("strategies_not_detected", [])
            times = sim.get("times", {})
            consensus_agreement = sim.get("consensus_agreement", 0.0)
            
            detected_sorted = sorted(detected) if detected else []
            not_detected_sorted = sorted(not_detected) if not_detected else []
            
            print(f"\n  User {user_id}:")
            print(f"    Strategies DETECTED liquidation: {', '.join(detected_sorted)} ({len(detected)}/6)")
            if detected and times:
                sorted_times = sorted(times.items())
                time_strs = [f"{name}: {t/(24*3600):.2f} days" for name, t in sorted_times]
                print(f"    Liquidation times: {', '.join(time_strs)}")
            if not_detected:
                print(f"    Strategies NOT detected: {', '.join(not_detected_sorted)} ({len(not_detected)}/6)")
            print(f"    Consensus agreement: {consensus_agreement:.1%}")
            
        if len(samples) > 10:
            print(f"\n  ... and {len(samples) - 10} more cases")

    disagreement_without = [s for s in per_sim_without if 0 < len(s.get("strategies_detected", [])) < 6]
    print_samples(disagreement_without, "Sample Cases with Strategy Disagreement (WITHOUT recommendation):")

    disagreement_with = [s for s in per_sim_with if 0 < len(s.get("strategies_detected", [])) < 6]
    print_samples(disagreement_with, "Sample Cases with Strategy Disagreement (WITH recommendation):")

    print("\nStrategy Agreement Statistics:")
    if per_sim_without:
        partial = len(disagreement_without)
        full = len(per_sim_without) - partial
        print(f"  WITHOUT recommendation:")
        print(f"    Full agreement: {full}/{len(per_sim_without)} ({full/len(per_sim_without):.1%})")
        print(f"    Partial agreement: {partial}/{len(per_sim_without)} ({partial/len(per_sim_without):.1%})")
    
    if per_sim_with:
        partial = len(disagreement_with)
        full = len(per_sim_with) - partial
        print(f"  WITH recommendation:")
        print(f"    Full agreement: {full}/{len(per_sim_with)} ({full/len(per_sim_with):.1%})")
        print(f"    Partial agreement: {partial}/{len(per_sim_with)} ({partial/len(per_sim_with):.1%})")

    def print_liquidations(samples, title):
        print(f"\n{title}")
        liquidated = [s for s in samples if s.get("strategies_detected")]
        if not liquidated:
            print("  No liquidations detected.")
            return
            
        for i, sim in enumerate(liquidated[:5]):
            user_id = sim.get("user", "unknown")
            detected = sim.get("strategies_detected", [])
            times = sim.get("times", {})
            checks = sim.get("checks", {})
            detected_sorted = sorted(detected) if detected else []
            
            print(f"\n  User {user_id}:")
            print(f"    Detected by: {', '.join(detected_sorted)} ({len(detected)}/6 strategies)")
            if times:
                print(f"    Liquidation times by strategy:")
                for strategy_name in detected_sorted:
                    if strategy_name in times:
                        time_days = times[strategy_name] / (24 * 3600)
                        check_count = checks.get(strategy_name, "N/A")
                        print(f"      - {strategy_name}: {time_days:.2f} days ({check_count} checks)")
            if len(detected) < 6:
                not_detected = sim.get("strategies_not_detected", [])
                print(f"    NOT detected by: {', '.join(sorted(not_detected))} ({len(not_detected)}/6 strategies)")
        
        if len(liquidated) > 5:
            print(f"\n  ... and {len(liquidated) - 5} more liquidations")

    print_liquidations(per_sim_without, "Sample Liquidations with Full Strategy Breakdown (WITHOUT recommendation):")
    print_liquidations(per_sim_with, "Sample Liquidations with Full Strategy Breakdown (WITH recommendation):")


def compare_statistics(stats, stats_no_dust):
    """Compare standard statistics with no-dust statistics."""
    print(f"\n{'='*80}")
    print("Standard vs No-Dust Comparison".center(80))
    print(f"{'='*80}")
    
    std_overall = stats.get('overall', {})
    nd_overall = stats_no_dust.get('overall', {})
    
    std_processed = std_overall.get('processed', 0)
    nd_processed = nd_overall.get('processed', 0)
    
    if std_processed == 0:
        print("No data in standard statistics.")
        return

    # Metrics to compare
    metrics = [
        ("Total Processed", "processed"),
        ("Liquidated WITHOUT", "liquidated_without"),
        ("Liquidated WITH", "liquidated_with"),
        ("Improved", "improved"),
        ("Worsened", "worsened"),
        ("No Change", "no_change")
    ]
    
    print(f"\n{'Metric':<30} {'Standard':<15} {'No-Dust':<15} {'Diff':<10}")
    print("-" * 75)
    
    for label, key in metrics:
        std_val = std_overall.get(key, 0)
        nd_val = nd_overall.get(key, 0)
        diff = nd_val - std_val
        print(f"{label:<30} {std_val:<15} {nd_val:<15} {diff:<10}")

    # Calculate rates
    def get_rate(s, key):
        p = s.get('processed', 0)
        return (s.get(key, 0) / p * 100) if p > 0 else 0.0

    print("-" * 75)
    print("Rates (% of processed):")
    
    rate_metrics = [
        ("Liquidation Rate (Without)", "liquidated_without"),
        ("Liquidation Rate (With)", "liquidated_with"),
        ("Improvement Rate", "improved"),
        ("Worsening Rate", "worsened")
    ]
    
    for label, key in rate_metrics:
        std_rate = get_rate(std_overall, key)
        nd_rate = get_rate(nd_overall, key)
        diff = nd_rate - std_rate
        print(f"{label:<30} {std_rate:>6.2f}%        {nd_rate:>6.2f}%        {diff:>+5.2f}%")


def create_visualizations(stats, output_dir):
    """Create visualization plots."""
    if not HAS_MATPLOTLIB:
        print("\nSkipping visualizations (matplotlib not available)")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    overall = stats.get('overall', {})
    
    # 1. Overall effectiveness comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Simulation Results Overview', fontsize=16, fontweight='bold')
    
    # Liquidation comparison
    ax = axes[0, 0]
    categories = ['Without\nRecommendation', 'With\nRecommendation']
    values = [overall.get('liquidated_without', 0), overall.get('liquidated_with', 0)]
    colors = ['#d62728', '#2ca02c']
    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel('Number of Liquidations')
    ax.set_title('Liquidation Count Comparison')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({val/overall.get("processed",1)*100:.1f}%)',
                ha='center', va='bottom')
    
    # Outcome distribution
    ax = axes[0, 1]
    outcomes = ['Improved', 'Worsened', 'No Change']
    values = [overall.get('improved', 0), overall.get('worsened', 0), overall.get('no_change', 0)]
    colors = ['#2ca02c', '#d62728', '#7f7f7f']
    ax.pie(values, labels=outcomes, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Recommendation Outcomes')
    
    # Liquidation reasons breakdown
    ax = axes[1, 0]
    reasons = ['Dust', 'HF-based', 'Threshold']
    without_vals = [
        overall.get('dust_liquidations_without', 0),
        overall.get('hf_based_liquidations_without', 0),
        overall.get('threshold_based_liquidations_without', 0),
    ]
    with_vals = [
        overall.get('dust_liquidations_with', 0),
        overall.get('hf_based_liquidations_with', 0),
        overall.get('threshold_based_liquidations_with', 0),
    ]
    
    x = np.arange(len(reasons))
    width = 0.35
    ax.bar(x - width/2, without_vals, width, label='Without', color='#d62728', alpha=0.7)
    ax.bar(x + width/2, with_vals, width, label='With', color='#2ca02c', alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Liquidation Reasons Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(reasons)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Strategy consensus agreement
    ax = axes[1, 1]
    strategy_comps = overall.get('strategy_comparisons', {})
    consensus_without = strategy_comps.get('consensus_agreement_without', [])
    consensus_with = strategy_comps.get('consensus_agreement_with', [])
    
    if consensus_without or consensus_with:
        data_to_plot = []
        labels = []
        if consensus_without:
            data_to_plot.append([x * 100 for x in consensus_without])
            labels.append('Without')
        if consensus_with:
            data_to_plot.append([x * 100 for x in consensus_with])
            labels.append('With')
        
        ax.boxplot(data_to_plot, labels=labels)
        ax.set_ylabel('Consensus Agreement (%)')
        ax.set_title('Strategy Consensus Agreement')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'simulation_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Action pair analysis
    by_pair = stats.get('by_action_pair', {})
    if by_pair:
        pair_metrics = []
        for pair_key, pair_stats in by_pair.items():
            processed = pair_stats.get('processed', 0)
            if processed == 0:
                continue
            improved = pair_stats.get('improved', 0)
            worsened = pair_stats.get('worsened', 0)
            pair_metrics.append({
                'pair': pair_key,
                'net_benefit': improved - worsened,
                'improvement_rate': (improved / processed * 100) if processed > 0 else 0,
            })
        
        pair_metrics.sort(key=lambda x: x['net_benefit'], reverse=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        top_pairs = pair_metrics[:15]
        pairs = [m['pair'] for m in top_pairs]
        net_benefits = [m['net_benefit'] for m in top_pairs]
        colors = ['#2ca02c' if b > 0 else '#d62728' if b < 0 else '#7f7f7f' for b in net_benefits]
        
        bars = ax.barh(range(len(pairs)), net_benefits, color=colors, alpha=0.7)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pairs)
        ax.set_xlabel('Net Benefit (Improved - Worsened)')
        ax.set_title('Top 15 Action Pairs by Net Benefit')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, net_benefits)):
            ax.text(val, bar.get_y() + bar.get_height()/2,
                   f'{val:+d}',
                   ha='left' if val > 0 else 'right', va='center')
        
        plt.tight_layout()
        fig.savefig(output_dir / 'action_pair_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # NEW: Risk Analysis Plots
    corr_data = overall.get("liquidation_prediction_correlation", [])
    times_tp = overall.get("at_risk_time_to_liquidation", [])
    times_fn = overall.get("not_at_risk_time_to_liquidation", [])

    if corr_data or (times_tp and times_fn):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Risk Prediction Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Time Distribution (TP vs FN)
        ax = axes[0]
        data_to_plot = []
        labels = []
        if times_tp:
            data_to_plot.append([t/(24*3600) for t in times_tp])
            labels.append('True Positives\n(Predicted)')
        if times_fn:
            data_to_plot.append([t/(24*3600) for t in times_fn])
            labels.append('False Negatives\n(Missed)')

        if data_to_plot:
            ax.boxplot(data_to_plot, labels=labels)
            ax.set_ylabel('Time to Liquidation (Days)')
            ax.set_title('Time to Liquidation Distribution')
            ax.set_yscale('log')
            ax.grid(axis='y', alpha=0.3)

        # Plot 2: Correlation
        ax = axes[1]
        if corr_data:
            predicted = [x['predicted'] / (24*3600) for x in corr_data]
            actual = [x['actual'] / (24*3600) for x in corr_data]

            ax.scatter(predicted, actual, alpha=0.5, s=15)
            ax.set_xlabel('Predicted Return Period (Days)')
            ax.set_ylabel('Actual Time to Liquidation (Days)')
            ax.set_title('Prediction vs Actual Correlation')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No correlation data available', ha='center', va='center')

        plt.tight_layout()
        fig.savefig(output_dir / 'risk_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n✓ Visualizations saved to: {output_dir}")


def export_detailed_analysis(stats, output_file):
    """Export detailed analysis to a text file."""
    output_path = Path(output_file)
    
    # Redirect stdout to file
    original_stdout = sys.stdout
    with open(output_path, 'w') as f:
        sys.stdout = f
        print("="*80)
        print("SIMULATION RESULTS ANALYSIS".center(80))
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        print_summary_table(stats)
        analyze_liquidation_reasons(stats)
        analyze_action_pairs(stats)
        analyze_strategy_comparison(stats)
        analyze_per_simulation_breakdown(stats)
        analyze_time_deltas(stats)
        analyze_risk_predictions(stats)
    
    sys.stdout = original_stdout
    print(f"✓ Detailed analysis exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze simulation results from performSimulations.py"
    )
    parser.add_argument(
        'json_file',
        type=str,
        help='Path to simulation statistics JSON file'
    )
    parser.add_argument(
        '--no-dust-file',
        type=str,
        default=None,
        help='Path to no-dust statistics JSON file (optional, for comparison)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./cache/analysis_output',
        help='Directory to save visualizations (default: ./cache/analysis_output)'
    )
    parser.add_argument(
        '--export-text',
        type=str,
        default=None,
        help='Export detailed text analysis to file'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating visualizations'
    )
    
    args = parser.parse_args()
    
    # Load statistics
    print(f"Loading statistics from: {args.json_file}")
    stats = load_statistics(args.json_file)
    print("✓ Statistics loaded")
    
    # Try to load no-dust stats
    stats_no_dust = None
    if args.no_dust_file:
        print(f"Loading no-dust statistics from: {args.no_dust_file}")
        stats_no_dust = load_statistics(args.no_dust_file)
    else:
        # Attempt auto-discovery based on naming convention
        try:
            p = Path(args.json_file)
            name = p.name
            # Standard: simulation_statistics_TIMESTAMP.json
            # No Dust: simulation_statistics_no_dust_TIMESTAMP.json
            if "simulation_statistics_" in name and "no_dust" not in name:
                parts = name.split("simulation_statistics_")
                if len(parts) == 2:
                    no_dust_name = f"simulation_statistics_no_dust_{parts[1]}"
                    no_dust_path = p.parent / no_dust_name
                    if no_dust_path.exists():
                        print(f"Auto-detected no-dust statistics: {no_dust_path}")
                        stats_no_dust = load_statistics(no_dust_path)
        except Exception as e:
            print(f"Auto-detection of no-dust file failed: {e}")

    # Print summary
    print_summary_table(stats)
    
    # Analyze liquidation reasons
    analyze_liquidation_reasons(stats)
    
    # Analyze action pairs
    analyze_action_pairs(stats)
    
    # Analyze strategy comparison
    analyze_strategy_comparison(stats)

    # Analyze per-simulation breakdown
    analyze_per_simulation_breakdown(stats)

    # Analyze time deltas
    analyze_time_deltas(stats)

    # Analyze risk predictions
    analyze_risk_predictions(stats)
    
    # Compare with no-dust if available
    if stats_no_dust:
        print("\n" + "="*80)
        print("NO-DUST STATISTICS DETAILED ANALYSIS")
        print("="*80)
        print_summary_table(stats_no_dust)
        analyze_liquidation_reasons(stats_no_dust)
        analyze_action_pairs(stats_no_dust)
        analyze_strategy_comparison(stats_no_dust)
        analyze_per_simulation_breakdown(stats_no_dust)
        analyze_time_deltas(stats_no_dust)
        analyze_risk_predictions(stats_no_dust)
        compare_statistics(stats, stats_no_dust)
    
    # Create visualizations
    if not args.no_plots:
        create_visualizations(stats, args.output_dir)
    
    # Export detailed analysis
    if args.export_text:
        export_detailed_analysis(stats, args.export_text)


if __name__ == '__main__':
    main()
