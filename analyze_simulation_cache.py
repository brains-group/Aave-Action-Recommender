#!/usr/bin/env python3
"""
Analyze cached simulation results to understand recommendation effectiveness.

This script examines individual simulation results to:
- Find successful vs failed recommendations
- Analyze recommendation amounts and their correlation with outcomes
- Identify patterns in dust liquidations
- Compare user states before/after recommendations
- Generate insights for improving the recommendation system
"""

import pickle
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_simulation_result(filepath):
    """Load a single simulation result pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_all_simulation_pairs(cache_dir):
    """Get pairs of (without, with) simulation results."""
    cache_path = Path(cache_dir)
    files = list(cache_path.glob("*.pkl"))
    
    # Group by user and timestamp
    pairs = defaultdict(dict)
    
    for filepath in files:
        filename = filepath.stem
        parts = filename.rsplit('_', 1)  # Split on last underscore
        if len(parts) == 2:
            user_timestamp, suffix = parts
            if suffix in ['with', 'without']:
                pairs[user_timestamp][suffix] = filepath
    
    # Only return pairs that have both 'with' and 'without'
    valid_pairs = []
    for key, files_dict in pairs.items():
        if 'with' in files_dict and 'without' in files_dict:
            valid_pairs.append({
                'key': key,
                'without': files_dict['without'],
                'with': files_dict['with']
            })
    
    return valid_pairs


def analyze_recommendation_outcome(pair):
    """Analyze the outcome of a recommendation by comparing with/without results."""
    try:
        without_data = load_simulation_result(pair['without'])
        with_data = load_simulation_result(pair['with'])
        
        liq_without = without_data.get('liquidation_stats', {}).get('liquidated', False)
        liq_with = with_data.get('liquidation_stats', {}).get('liquidated', False)
        
        outcome = 'no_change'
        if not liq_without and liq_with:
            outcome = 'worsened'
        elif liq_without and not liq_with:
            outcome = 'improved'
        elif liq_without and liq_with:
            # Both liquidated, check time difference
            t_without = without_data.get('liquidation_stats', {}).get('time_to_liquidation')
            t_with = with_data.get('liquidation_stats', {}).get('time_to_liquidation')
            if t_without and t_with:
                if t_with > t_without:
                    outcome = 'improved_delay'
                elif t_with < t_without:
                    outcome = 'worsened_accelerated'
            else:
                outcome = 'both_liquidated'
        
        return {
            'key': pair['key'],
            'user': with_data.get('user_address', 'unknown'),
            'outcome': outcome,
            'liq_without': liq_without,
            'liq_with': liq_with,
            'final_state_without': without_data.get('final_state', {}),
            'final_state_with': with_data.get('final_state', {}),
            'liquidation_stats_without': without_data.get('liquidation_stats', {}),
            'liquidation_stats_with': with_data.get('liquidation_stats', {}),
            'num_transactions': with_data.get('num_transactions', 0),
        }
    except Exception as e:
        print(f"Error processing pair {pair['key']}: {e}")
        return None


def analyze_all_pairs(cache_dir):
    """Analyze all simulation pairs and return aggregated statistics."""
    pairs = get_all_simulation_pairs(cache_dir)
    print(f"Found {len(pairs)} simulation pairs")
    
    results = []
    for pair in pairs:
        result = analyze_recommendation_outcome(pair)
        if result:
            results.append(result)
    
    return results


def find_dust_liquidation_cases(results):
    """Find cases where dust liquidations occurred."""
    dust_cases = []
    
    for result in results:
        if result['liq_with']:
            reason = result['liquidation_stats_with'].get('liquidation_reason', '')
            if 'dust' in reason.lower():
                dust_cases.append(result)
        elif result['liq_without']:
            reason = result['liquidation_stats_without'].get('liquidation_reason', '')
            if 'dust' in reason.lower():
                dust_cases.append(result)
    
    return dust_cases


def filter_dust_results(results):
    """Filter out results where dust liquidation occurred in either branch."""
    filtered = []
    for r in results:
        is_dust = False
        # Check without recommendation
        if r['liq_without']:
            reason = r['liquidation_stats_without'].get('liquidation_reason', '')
            if 'dust' in reason.lower():
                is_dust = True
        
        # Check with recommendation
        if r['liq_with']:
            reason = r['liquidation_stats_with'].get('liquidation_reason', '')
            if 'dust' in reason.lower():
                is_dust = True
                
        if not is_dust:
            filtered.append(r)
    return filtered


def analyze_final_states(results):
    """Analyze final states to understand position sizes."""
    analysis = {
        'improved': [],
        'worsened': [],
        'no_change': [],
        'dust_liquidations': [],
    }
    
    for result in results:
        outcome = result['outcome']
        final_with = result['final_state_with']
        
        # Extract position metrics
        total_debt = final_with.get('total_debt_usd', 0)
        total_collateral = final_with.get('total_collateral_usd', 0)
        hf = final_with.get('health_factor', None)
        
        metrics = {
            'total_debt_usd': total_debt,
            'total_collateral_usd': total_collateral,
            'health_factor': hf,
            'user': result['user'],
            'key': result['key'],
        }
        
        # Check if dust liquidation
        liq_stats = result['liquidation_stats_with']
        if result['liq_with']:
            reason = liq_stats.get('liquidation_reason', '')
            if 'dust' in reason.lower():
                analysis['dust_liquidations'].append(metrics)
        
        # Categorize by outcome
        if outcome in analysis:
            analysis[outcome].append(metrics)
    
    return analysis


def print_summary_statistics(results, title="SIMULATION CACHE ANALYSIS SUMMARY"):
    """Print summary statistics from analysis."""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    total = len(results)
    outcomes = Counter(r['outcome'] for r in results)
    
    print(f"\nTotal simulation pairs analyzed: {total}")
    print("\nOutcome Distribution:")
    for outcome, count in outcomes.most_common():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {outcome:20s}: {count:5d} ({pct:5.2f}%)")
    
    # Liquidation statistics
    liq_without = sum(1 for r in results if r['liq_without'])
    liq_with = sum(1 for r in results if r['liq_with'])
    improved = sum(1 for r in results if r['outcome'] == 'improved')
    worsened = sum(1 for r in results if r['outcome'] == 'worsened')
    
    print(f"\nLiquidation Statistics:")
    print(f"  Liquidated WITHOUT recommendation: {liq_without} ({liq_without/total*100:.2f}%)")
    print(f"  Liquidated WITH recommendation:    {liq_with} ({liq_with/total*100:.2f}%)")
    print(f"  Improved (avoided liquidation):     {improved} ({improved/total*100:.2f}%)")
    print(f"  Worsened (introduced liquidation):  {worsened} ({worsened/total*100:.2f}%)")
    
    # Dust liquidation analysis
    dust_cases = find_dust_liquidation_cases(results)
    print(f"\nDust Liquidation Cases: {len(dust_cases)}")
    
    # Analyze final states
    state_analysis = analyze_final_states(results)
    
    print(f"\nFinal State Analysis:")
    for category in ['improved', 'worsened', 'dust_liquidations']:
        cases = state_analysis[category]
        if cases:
            debts = [c['total_debt_usd'] for c in cases if c['total_debt_usd'] > 0]
            collaterals = [c['total_collateral_usd'] for c in cases if c['total_collateral_usd'] > 0]
            
            print(f"\n  {category.upper()} ({len(cases)} cases):")
            if debts:
                print(f"    Debt USD:      min={min(debts):.4f}, max={max(debts):.2f}, median={np.median(debts):.2f}")
            if collaterals:
                print(f"    Collateral USD: min={min(collaterals):.4f}, max={max(collaterals):.2f}, median={np.median(collaterals):.2f}")


def find_successful_patterns(results):
    """Find patterns in successful recommendations."""
    improved = [r for r in results if r['outcome'] == 'improved']
    worsened = [r for r in results if r['outcome'] == 'worsened']
    
    print("\n" + "="*80)
    print("SUCCESSFUL PATTERNS ANALYSIS")
    print("="*80)
    
    print(f"\nImproved Cases: {len(improved)}")
    if improved:
        print("Sample improved cases:")
        for i, case in enumerate(improved[:5]):
            final = case['final_state_with']
            liq_stats = case['liquidation_stats_with']
            print(f"\n  Case {i+1}: {case['user'][:10]}...")
            print(f"    Final HF: {final.get('health_factor', 'N/A')}")
            print(f"    Final Debt: ${final.get('total_debt_usd', 0):.2f}")
            print(f"    Final Collateral: ${final.get('total_collateral_usd', 0):.2f}")
    
    print(f"\nWorsened Cases: {len(worsened)}")
    if worsened:
        print("Sample worsened cases:")
        for i, case in enumerate(worsened[:5]):
            final = case['final_state_with']
            liq_stats = case['liquidation_stats_with']
            reason = liq_stats.get('liquidation_reason', 'N/A')
            print(f"\n  Case {i+1}: {case['user'][:10]}...")
            print(f"    Liquidation reason: {reason[:80]}")
            print(f"    Final Debt: ${final.get('total_debt_usd', 0):.2f}")
            print(f"    Final Collateral: ${final.get('total_collateral_usd', 0):.2f}")


def export_detailed_cases(results, output_file):
    """Export detailed case-by-case analysis to JSON."""
    export_data = []
    
    for result in results:
        export_data.append({
            'user': result['user'],
            'key': result['key'],
            'outcome': result['outcome'],
            'liq_without': result['liq_without'],
            'liq_with': result['liq_with'],
            'final_state_without': {
                'health_factor': result['final_state_without'].get('health_factor'),
                'total_debt_usd': result['final_state_without'].get('total_debt_usd'),
                'total_collateral_usd': result['final_state_without'].get('total_collateral_usd'),
            },
            'final_state_with': {
                'health_factor': result['final_state_with'].get('health_factor'),
                'total_debt_usd': result['final_state_with'].get('total_debt_usd'),
                'total_collateral_usd': result['final_state_with'].get('total_collateral_usd'),
            },
            'liquidation_reason_without': result['liquidation_stats_without'].get('liquidation_reason'),
            'liquidation_reason_with': result['liquidation_stats_with'].get('liquidation_reason'),
        })
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n✓ Detailed cases exported to: {output_file}")


def create_visualizations(results, output_dir):
    """Create visualization plots from analysis."""
    if not HAS_MATPLOTLIB:
        print("\nSkipping visualizations (matplotlib not available)")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    state_analysis = analyze_final_states(results)
    
    # Plot 1: Position sizes for different outcomes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Final Position Sizes by Outcome', fontsize=16, fontweight='bold')
    
    # Debt distribution
    ax = axes[0, 0]
    categories = ['Improved', 'Worsened', 'Dust Liquidations']
    data_to_plot = []
    labels = []
    for cat in categories:
        key = cat.lower().replace(' ', '_')
        cases = state_analysis.get(key, [])
        if cases:
            debts = [c['total_debt_usd'] for c in cases if c['total_debt_usd'] > 0]
            if debts:
                data_to_plot.append(debts)
                labels.append(cat)
    
    if data_to_plot:
        ax.boxplot(data_to_plot, labels=labels)
        ax.set_ylabel('Total Debt (USD)')
        ax.set_title('Debt Distribution by Outcome')
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)
    
    # Collateral distribution
    ax = axes[0, 1]
    data_to_plot = []
    labels = []
    for cat in categories:
        key = cat.lower().replace(' ', '_')
        cases = state_analysis.get(key, [])
        if cases:
            collaterals = [c['total_collateral_usd'] for c in cases if c['total_collateral_usd'] > 0]
            if collaterals:
                data_to_plot.append(collaterals)
                labels.append(cat)
    
    if data_to_plot:
        ax.boxplot(data_to_plot, labels=labels)
        ax.set_ylabel('Total Collateral (USD)')
        ax.set_title('Collateral Distribution by Outcome')
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)
    
    # Outcome distribution pie chart
    ax = axes[1, 0]
    outcomes = Counter(r['outcome'] for r in results)
    if outcomes:
        labels_pie = list(outcomes.keys())
        sizes = list(outcomes.values())
        ax.pie(sizes, labels=labels_pie, autopct='%1.1f%%', startangle=90)
        ax.set_title('Outcome Distribution')
    
    # Liquidation comparison
    ax = axes[1, 1]
    liq_without = sum(1 for r in results if r['liq_without'])
    liq_with = sum(1 for r in results if r['liq_with'])
    non_liq_without = len(results) - liq_without
    non_liq_with = len(results) - liq_with
    
    categories = ['Without\nRecommendation', 'With\nRecommendation']
    liquidated = [liq_without, liq_with]
    not_liquidated = [non_liq_without, non_liq_with]
    
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, liquidated, width, label='Liquidated', color='#d62728', alpha=0.7)
    ax.bar(x + width/2, not_liquidated, width, label='Not Liquidated', color='#2ca02c', alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Liquidation Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'simulation_cache_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cached simulation results for recommendation insights"
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./cache/simulation_results',
        help='Directory containing cached simulation results (default: ./cache/simulation_results)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./cache/analysis_output',
        help='Directory to save output files (default: ./cache/analysis_output)'
    )
    parser.add_argument(
        '--export-json',
        type=str,
        default=None,
        help='Export detailed case analysis to JSON file'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating visualizations'
    )
    
    args = parser.parse_args()
    
    # Analyze all simulation pairs
    print(f"Analyzing simulation results from: {args.cache_dir}")
    results = analyze_all_pairs(args.cache_dir)
    
    if not results:
        print("No valid simulation pairs found!")
        return
    
    # Print summary statistics
    print_summary_statistics(results, "FULL SIMULATION RESULTS (Including Dust)")
    
    # Filter dust and print summary
    results_no_dust = filter_dust_results(results)
    if len(results_no_dust) < len(results):
        print_summary_statistics(results_no_dust, "FILTERED RESULTS (Excluding Dust Liquidations)")
    else:
        print("\nNo dust liquidations found to filter.")
    
    # Find successful patterns
    find_successful_patterns(results)
    
    # Export detailed cases if requested
    if args.export_json:
        export_detailed_cases(results, args.export_json)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        export_detailed_cases(results, output_dir / 'detailed_cases.json')
    
    # Create visualizations
    if not args.no_plots:
        create_visualizations(results, args.output_dir)


if __name__ == '__main__':
    main()
