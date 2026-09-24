"""Fixed-horizon metrics with explicit censoring exclusions (not IPCW estimates)."""
import math

def horizon_metrics(rows, horizon, threshold=.5):
    from sklearn.metrics import roc_auc_score, average_precision_score
    if not math.isfinite(horizon) or horizon<=0 or not 0<=threshold<=1:raise ValueError('Invalid horizon/threshold')
    kept=[];excluded=[];seen=set()
    for r in rows:
        if r['case_id'] in seen:raise ValueError('Duplicate case ID')
        seen.add(r['case_id'])
        t=float(r['duration']);p=float(r['probability']);event=r['event']
        if event not in (0,1) or not math.isfinite(t) or t<0 or not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid prediction/label')
        if r['horizon']!=horizon:raise ValueError('Prediction horizon mismatch')
        if not event and t<horizon:excluded.append(r['case_id']);continue
        kept.append((int(bool(event) and t<=horizon),p))
    n=len(kept);tp=sum(y and p>=threshold for y,p in kept);fp=sum(not y and p>=threshold for y,p in kept)
    fn=sum(y and p<threshold for y,p in kept);tn=n-tp-fp-fn
    ratio=lambda a,b:a/b if b else None
    y=[a for a,p in kept];p=[p for a,p in kept]
    calibration=[]
    for i in range(10):
        group=[(a,p) for a,p in kept if min(9,int(p*10))==i]
        calibration.append(dict(lower=i/10,upper=(i+1)/10,n=len(group),mean_prediction=ratio(sum(p for a,p in group),len(group)),event_fraction=ratio(sum(a for a,p in group),len(group))))
    return dict(horizon=horizon,threshold=threshold,n=n,excluded_censored_ids=excluded,
                tp=tp,fp=fp,tn=tn,fn=fn,prevalence=ratio(sum(y),n),accuracy=ratio(tp+tn,n),
                precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),fpr=ratio(fp,fp+tn),
                roc_auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
                average_precision=float(average_precision_score(y,p)) if len(set(y))==2 else None,
                brier=ratio(sum((p-a)**2 for a,p in kept),n),calibration=calibration,
                estimand='known-horizon-label subset; no censoring weighting; not a population survival estimate')
