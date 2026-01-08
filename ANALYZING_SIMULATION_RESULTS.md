# Analyzing Cached Simulation Results

This guide explains how to use the cached simulation results in `cache/simulation_results/` to understand and improve the recommendation system.

## Overview

Each simulation result is stored as a pickle file with the naming pattern:
```
{user_address}_{timestamp}_{suffix}.pkl
```
Where `suffix` is either `with` or `without` (recommendation).

## Data Structure

Each pickle file contains a dictionary with:
- `user_address`: User wallet address
- `final_state`: Final user state after simulation
  - `health_factor`: Final health factor
  - `total_debt_usd`: Total debt in USD
  - `total_collateral_usd`: Total collateral in USD
  - `available_borrows_usd`: Available borrow capacity
  - `wallet_balances`: Wallet balances by asset
  - `collateral`: Collateral amounts by asset
  - `debt`: Debt amounts by asset
- `num_transactions`: Number of transactions processed
- `liquidation_stats`: Liquidation detection results
  - `liquidated`: Boolean indicating if liquidation occurred
  - `time_to_liquidation`: Seconds until liquidation (if liquidated)
  - `liquidation_reason`: String describing why liquidation occurred
  - `strategy_comparison`: Detailed results from all 6 strategies
  - `best_strategy`: Most efficient strategy that detected liquidation
  - `consensus_agreement`: Agreement rate across strategies

## Analysis Tools

### 1. Comprehensive Cache Analysis

```bash
python3 analyze_simulation_cache.py
```

**What it does:**
- Analyzes all simulation pairs (with/without recommendation)
- Compares outcomes (improved, worsened, no change)
- Identifies dust liquidation cases
- Analyzes final position sizes
- Generates visualizations

**Outputs:**
- Console summary statistics
- `analysis_output/detailed_cases.json`: Case-by-case analysis
- `analysis_output/simulation_cache_analysis.png`: Visualization charts

**Key Insights from Current Analysis:**
- **7,085 simulation pairs analyzed**
- **56 improved cases (0.79%)** - Recommendations that successfully prevented liquidation
- **250 worsened cases (3.53%)** - Recommendations that caused liquidation
- **3,673 dust liquidation cases** - Majority of liquidations are tiny positions
- **Dust liquidations have median debt of $0.01** - Recommendations creating extremely small positions

### 2. Individual Case Investigation

To examine a specific case:

```python
import pickle

# Load a specific simulation result
with open('cache/simulation_results/0x..._timestamp_with.pkl', 'rb') as f:
    result = pickle.load(f)

print(f"User: {result['user_address']}")
print(f"Liquidated: {result['liquidation_stats']['liquidated']}")
print(f"Final HF: {result['final_state']['health_factor']}")
print(f"Final Debt: ${result['final_state']['total_debt_usd']:.2f}")
print(f"Liquidation Reason: {result['liquidation_stats']['liquidation_reason']}")
```

### 3. Finding Successful Patterns

The analysis script identifies successful cases. Key patterns from current data:

**Improved Cases Characteristics:**
- Median debt: $2,713 (moderate positions)
- Median collateral: $32,371 (substantial collateral)
- Final HF ranges: 2.15 - 21.35 (healthy positions)

**Worsened Cases Characteristics:**
- Often have HF just below threshold (1.12 - 1.46)
- Recommendations may push HF over edge
- Time gaps to liquidation: 47-167 days

**Dust Liquidation Cases:**
- Median debt: $0.01 (extremely small)
- Median collateral: $6,372 (moderate, but debt tiny)
- These are positions so small they get liquidated immediately

## Use Cases

### 1. Debugging Recommendation Failures

**Find cases where recommendations failed:**
```python
import pickle
import glob

# Find all "worsened" cases
for filepath in glob.glob('cache/simulation_results/*_with.pkl'):
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
    
    # Find corresponding "without" file
    without_file = filepath.replace('_with.pkl', '_without.pkl')
    if Path(without_file).exists():
        with open(without_file, 'rb') as f:
            without_result = pickle.load(f)
        
        # Check if recommendation made things worse
        if not without_result['liquidation_stats']['liquidated'] and \
           result['liquidation_stats']['liquidated']:
            print(f"Worsened: {result['user_address']}")
            print(f"  Reason: {result['liquidation_stats']['liquidation_reason']}")
```

### 2. Understanding Amount Optimization

The cached results don't directly contain recommendation amounts, but you can:
1. Compare final states to infer recommendation impact
2. Look at position size changes
3. Analyze which recommendations created dust positions

**Example: Analyze dust liquidation creation**
```python
# Find cases where recommendation created dust liquidation
for filepath in glob.glob('cache/simulation_results/*_with.pkl'):
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
    
    liq_stats = result['liquidation_stats']
    if liq_stats['liquidated'] and 'dust' in liq_stats['liquidation_reason'].lower():
        final = result['final_state']
        debt = final['total_debt_usd']
        collateral = final['total_collateral_usd']
        
        if debt < 1.0:  # Dust liquidation
            print(f"User: {result['user_address']}")
            print(f"  Dust debt: ${debt:.6f}")
            print(f"  Collateral: ${collateral:.2f}")
            # This recommendation created a tiny position that got liquidated
```

### 3. Strategy Performance Analysis

Each result contains `strategy_comparison` data:
```python
result = pickle.load(open('cache/simulation_results/..._with.pkl', 'rb'))
strategy_comp = result['liquidation_stats']['strategy_comparison']

# See which strategy was most efficient
print(f"Best strategy: {result['liquidation_stats']['best_strategy']}")

# See consensus agreement
print(f"Consensus: {strategy_comp['consensus']['agreement_rate']:.1%}")

# See individual strategy results
for strategy_name, strategy_result in strategy_comp['results_by_strategy'].items():
    print(f"{strategy_name}: {strategy_result['checks_performed']} checks")
```

### 4. Recommendation Timing Analysis

Compare `time_to_liquidation` between with/without:
```python
# Load pair
with open('cache/simulation_results/..._without.pkl', 'rb') as f:
    without = pickle.load(f)
with open('cache/simulation_results/..._with.pkl', 'rb') as f:
    with_rec = pickle.load(f)

# Compare liquidation timing
if without['liquidation_stats']['liquidated'] and with_rec['liquidation_stats']['liquidated']:
    t_without = without['liquidation_stats']['time_to_liquidation']
    t_with = with_rec['liquidation_stats']['time_to_liquidation']
    
    if t_with > t_without:
        print(f"Recommendation delayed liquidation by {t_with - t_without:.0f} seconds")
    elif t_with < t_without:
        print(f"Recommendation accelerated liquidation by {t_without - t_with:.0f} seconds")
```

## Key Findings from Current Cache

### Critical Issues Identified:

1. **Dust Liquidation Problem**
   - 3,673 out of 4,360 liquidations (84.2%) are dust liquidations
   - Median dust debt: $0.01
   - This suggests recommendations are creating extremely small positions
   - **Recommendation**: Add minimum amount thresholds before suggesting actions

2. **Low Improvement Rate**
   - Only 56 cases (0.79%) improved
   - 250 cases (3.53%) worsened
   - **Recommendation**: Review recommendation logic, especially amount calculation

3. **Position Size Patterns**
   - Improved cases: Moderate debt ($2,713 median), substantial collateral ($32,371 median)
   - Worsened cases: Similar sizes, but recommendations pushed HF below threshold
   - **Recommendation**: Recommendations may need larger amounts or better timing

4. **Strategy Consensus**
   - 100% consensus agreement across strategies
   - Event-driven strategy most efficient
   - Detection framework is working correctly

## Actionable Recommendations

Based on cache analysis:

### Immediate Actions:

1. **Filter Dust Recommendations**
   - Don't recommend actions that would create positions < $10
   - Check final position size before recommending

2. **Increase Recommendation Amounts**
   - Current fixed amount of 10 is likely too small
   - Calculate amounts based on:
     - Required HF improvement
     - User's available balance
     - Minimum viable position size

3. **Improve Timing**
   - Most recommendations have no effect (95.72%)
   - Recommendations may be too late
   - Consider earlier intervention

4. **Target Successful Patterns**
   - Focus on users with moderate debt ($1K-$50K)
   - Ensure recommendations result in healthy HF (>2.0)
   - Avoid recommendations for already-liquidated users

## Exporting Data for Further Analysis

Export to JSON for custom analysis:
```bash
python3 analyze_simulation_cache.py --export-json custom_analysis.json
```

Then use in Python:
```python
import json

with open('custom_analysis.json', 'r') as f:
    cases = json.load(f)

# Filter for specific patterns
improved = [c for c in cases if c['outcome'] == 'improved']
worsened = [c for c in cases if c['outcome'] == 'worsened']

# Analyze patterns
for case in improved[:10]:
    print(f"User: {case['user']}")
    print(f"  Final HF: {case['final_state_with']['health_factor']}")
    print(f"  Final Debt: ${case['final_state_with']['total_debt_usd']:.2f}")
```

## Next Steps

1. **Deep Dive on Improved Cases**: Analyze the 56 successful cases to identify common patterns
2. **Analyze Recommendation Amounts**: Extract recommendation amounts from original data and correlate with outcomes
3. **Timing Analysis**: Determine optimal timing for recommendations (how early before liquidation?)
4. **Dust Prevention**: Develop filters to prevent dust position creation
5. **A/B Testing Framework**: Use cache to test new recommendation strategies

## Script Usage

```bash
# Basic analysis
python3 analyze_simulation_cache.py

# Custom output directory
python3 analyze_simulation_cache.py --output-dir ./my_analysis

# Skip visualizations
python3 analyze_simulation_cache.py --no-plots

# Custom JSON export
python3 analyze_simulation_cache.py --export-json ./results.json

# Custom cache directory
python3 analyze_simulation_cache.py --cache-dir ./custom/cache/path
```
