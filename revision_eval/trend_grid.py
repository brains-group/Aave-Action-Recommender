"""Exact finite-grid sensitivity from cached prediction histories; no model reruns."""
import argparse,pickle,json
from pathlib import Path
from itertools import product
import numpy as np

def polynomial_scores(score,grid):
 a=score(0,0,0);b=score(1,0,0)-a;c=score(0,1,0)-a;d=score(0,0,1)-a
 e=score(1,0,1)-a-b-d;f=score(0,1,1)-a-c-d
 w,l,g=np.asarray(grid).T
 return a+b*w+c*l+d*g+e*w*g+f*l*g

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
 import actionAgentTraining as agent
 grid=np.array(list(product(np.linspace(.6,1.,21),np.linspace(.4,.8,21),np.linspace(.1,.5,21))));base=np.array([.8,.6,.3]);dist=np.linalg.norm(grid-base,axis=1);maximum=dist.max();rows=[]
 for source in sorted((a.run/'cases').glob('*-core/recommendations.pkl')):
  pack=pickle.load(source.open('rb'));history=pack.get('prediction_history')
  if not history:rows.append(dict(case_id=pack['case_id'],error='prediction history unavailable'));continue
  last=history[max(history)];events=list(last)
  try:
   if any(v is None or not np.isfinite(v) for v in last.values()):raise ValueError('Incomplete event predictions')
   immediate=last['Liquidated']<=min(last.values());scores=[];base_scores=[]
   for event in events:
    data={ts:x[event] for ts,x in history.items() if event in x}
    score=lambda w,l,g:agent.calculate_trend_slope(data,w,l,g)
    scores.append(polynomial_scores(score,grid));base_scores.append(score(*base))
   scores=np.array(scores);i=events.index('Liquidated');flags=np.ones(len(grid),dtype=bool) if immediate else (scores[i]<0)&(scores[i]<=scores.min(axis=0))
   original=immediate or (base_scores[i]<0 and base_scores[i]<=min(base_scores));flips=flags!=original
   minimum=float(dist[flips].min()) if flips.any() else None
   groups=[flips[np.round(dist,4)==d].any() for d in sorted(set(np.round(dist,4))) if d>0]
   nonmonotonic=any(x and not y for x,y in zip(groups,groups[1:]))
   rows.append(dict(case_id=pack['case_id'],cohort=pack['cohort'],base_at_risk=bool(original),min_distance=minimum,score=minimum/maximum if minimum is not None else 1.,nonmonotonic_group_predicate=bool(nonmonotonic),grid_points=len(grid)))
  except Exception as e:rows.append(dict(case_id=pack['case_id'],error=str(e)))
 result={'rows':rows,'evaluated':sum('score' in r for r in rows),'errors':sum('error' in r for r in rows),'stability_score':float(np.mean([r['score'] for r in rows if 'score' in r])) if any('score' in r for r in rows) else None,'scope':'new disjoint cohorts, cached refreshed probabilities; not a reproduction of paper cohort 0.9353'}
 (a.run/'trend-sensitivity.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='rows'})
if __name__=='__main__':main()
