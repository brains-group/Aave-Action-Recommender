"""Right-continuous Breslow horizon probabilities with explicit extrapolation."""
import numpy as np

def horizon_probability(log_margin, baseline, horizon=604800.):
    times=np.asarray(baseline['times'],dtype=float);hazard=np.asarray(baseline['cum_hazards'],dtype=float)
    if horizon<=0 or not np.isfinite(horizon) or len(times)==0 or len(times)!=len(hazard) or np.any(np.diff(times)<0) or np.any(np.diff(hazard)<0) or np.any(hazard<0):raise ValueError('Invalid baseline/horizon')
    i=np.searchsorted(times,horizon,side='right')-1
    h0=0. if i<0 else hazard[i]
    if horizon>baseline['max_time']:h0=hazard[-1]+(horizon-baseline['max_time'])*baseline['final_rate']
    relative=np.exp(np.clip(np.asarray(log_margin)-baseline['log_shift'],-20,20))
    return -np.expm1(np.clip(-h0*relative,-50,0))
