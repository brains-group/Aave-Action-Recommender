"""Bounded chronological KM, ridge-Cox and XGBoost-Cox comparison.

Fit preprocessing and censoring weights on training data only. All tasks retain
source labels; zero durations receive a documented 1e-6-second numerical floor.
"""
import argparse,json,hashlib,pickle,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import horizon_metrics
from .probability import horizon_probability

def select(frame,n):
 ids=frame[['user','timestamp','pool','Index Event']].astype(str).agg(':'.join,axis=1)
 scores=pd.util.hash_pandas_object(ids,index=False);return frame.loc[scores.nsmallest(min(n,len(frame))).index]
def baseline(margin,duration,event):
 shift=float(np.median(margin));risk=np.exp(np.clip(margin-shift,-20,20))
 f=pd.DataFrame(dict(time=duration,event=event,risk=risk)).groupby('time').sum().sort_index()
 sums=np.cumsum(f.risk.to_numpy()[::-1])[::-1];cum=np.cumsum(f.event.to_numpy()/sums);times=f.index.to_numpy()
 return dict(times=times,cum_hazards=cum,max_time=float(times[-1]),final_rate=0.,log_shift=shift)
def main():
 from sklearn.compose import ColumnTransformer
 from sklearn.pipeline import make_pipeline
 from sklearn.impute import SimpleImputer
 from sklearn.preprocessing import StandardScaler,OneHotEncoder
 from sklearn.feature_selection import VarianceThreshold
 from sksurv.linear_model import CoxPHSurvivalAnalysis
 from sksurv.nonparametric import kaplan_meier_estimator
 from sksurv.metrics import concordance_index_censored,concordance_index_ipcw,cumulative_dynamic_auc,brier_score,integrated_brier_score
 from sksurv.util import Surv
 import xgboost as xgb
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cutoff',type=int,required=True);p.add_argument('--end',type=int,required=True);p.add_argument('--tasks',nargs='+');p.add_argument('--train-size',type=int,default=10000);p.add_argument('--test-size',type=int,default=2000);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 report={};horizons=np.array([86400.,259200.,604800.])
 for source in sorted(a.data.glob('*/Liquidated/data.csv')):
  task=source.parent.parent.name
  if a.tasks and task not in a.tasks:continue
  train=pd.DataFrame();test=pd.DataFrame()
  for c in pd.read_csv(source,chunksize=100000):
   train=select(pd.concat([train,c[c.timestamp<a.cutoff]],ignore_index=True),a.train_size)
   test=select(pd.concat([test,c[(c.timestamp>=a.cutoff)&(c.timestamp+604800<=a.end)]],ignore_index=True),a.test_size)
  result={'train_n':len(train),'test_n':len(test),'cutoff':a.cutoff,'source':str(source),'models':{}}
  if not len(train) or not len(test):
   result['unavailable']='No source observations for this task/window';report[task]=result;(a.output/'results.json').write_text(json.dumps(report,indent=2));continue
  try:
   for f,limit in [(train,a.cutoff),(test,a.end)]:
    available=limit-f.timestamp.to_numpy();f['status']=(f.status.astype(bool)&(f.timeDiff<=available)).astype(int);f['timeDiff']=np.maximum(1e-6,np.minimum(f.timeDiff,available))
   drop=['user','pool','id','observation_id','outcome_id','timestamp','timeDiff','status','split','Outcome Event']
   x=train.drop(columns=drop,errors='ignore');xt=test.reindex(columns=x.columns)
   numeric=x.select_dtypes(include=[np.number,'bool']).columns.tolist();cats=[c for c in x if c not in numeric]
   prep=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),numeric),('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False,max_categories=20)),cats)])
   x=prep.fit_transform(x);xt=prep.transform(xt);sel=VarianceThreshold();x=sel.fit_transform(x);xt=sel.transform(xt)
   y=Surv.from_arrays(train.status.astype(bool),train.timeDiff);yt=Surv.from_arrays(test.status.astype(bool),test.timeDiff)
   estimators={}
   kt,ks=kaplan_meier_estimator(y['event'],y['time']);survival=np.array([ks[max(0,np.searchsorted(kt,t,side='right')-1)] if t>=kt[0] else 1 for t in horizons]);estimators['KM']=(np.zeros(len(test)),np.tile(survival,(len(test),1)))
   solver_log=[];converged=False
   # Numerical fallback uses only convergence, never test outcomes.
   for penalty in [1.,10.,100.]:
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always')
     cox=CoxPHSurvivalAnalysis(alpha=penalty,n_iter=1000).fit(x,y)
    solver_log.append(dict(alpha=penalty,warnings=sorted(set(str(w.message) for w in caught))))
    if not any(w.category.__name__=='ConvergenceWarning' for w in caught):
     converged=True;break
   result['cox_solver_attempts']=solver_log
   if not converged:
    result['models']['CoxPH']={'unavailable':'Solver did not converge; no metrics claimed'}
   else:
    risk=cox.predict(xt);sf=cox.predict_survival_function(xt)
    estimators['CoxPH']=(risk,np.array([[f(t) for t in horizons] for f in sf]))
   matrix=xgb.DMatrix(x,label=np.where(y['event'],y['time'],-y['time']));model=xgb.train(dict(objective='survival:cox',tree_method='hist',max_depth=3,eta=.05,seed=42,nthread=1),matrix,num_boost_round=200)
   meta=baseline(model.predict(matrix,output_margin=True),y['time'],y['event']);margin=model.predict(xgb.DMatrix(xt),output_margin=True)
   estimators['XGBoostCox']=(margin,np.column_stack([1-horizon_probability(margin,meta,t) for t in horizons]))
   for name,(risk,survival) in estimators.items():
    metrics={'harrell_c':float(concordance_index_censored(yt['event'],yt['time'],risk)[0])}
    for metric,call in [('uno_c_7d',lambda:concordance_index_ipcw(y,yt,risk,tau=604800)[0]),('dynamic_auc',lambda:cumulative_dynamic_auc(y,yt,risk,horizons)[0].tolist()),('ipcw_brier',lambda:brier_score(y,yt,survival,horizons)[1].tolist()),('integrated_brier_1_to_7d',lambda:integrated_brier_score(y,yt,survival,horizons))]:
     try:metrics[metric]=call()
     except ValueError as e:metrics[metric]={'unavailable':str(e)}
    predictions=[dict(case_id=f'{task}:{i}',duration=float(t),event=int(e),probability=float(1-s),horizon=604800) for i,(t,e,s) in enumerate(zip(yt['time'],yt['event'],survival[:,-1]))]
    metrics['known_label_7d']=horizon_metrics(predictions,604800);result['models'][name]=metrics
    pd.DataFrame(predictions).assign(user=test.user.to_numpy(),timestamp=test.timestamp.to_numpy(),pool=test.pool.to_numpy()).to_csv(a.output/(task+'-'+name+'-predictions.csv'),index=False)
   result['features']=prep.get_feature_names_out().tolist();result['training_account_overlap']=len(set(train.user)&set(test.user));result['numerical_zero_duration_floor']=1e-6
   with (a.output/(task+'-artifacts.pkl')).open('wb') as f:pickle.dump(dict(preprocessor=prep,selector=sel,cox=cox,xgboost=model,baseline=meta),f)
  except Exception as e:result['error']=repr(e)
  report[task]=result;(a.output/'results.json').write_text(json.dumps(report,indent=2));print(task,'complete' if 'error' not in result else result['error'],flush=True)
 (a.output/'complete.json').write_text(json.dumps({'tasks':len(report),'errors':sum('error' in r for r in report.values()),'configuration':'10k/2k deterministic samples per start type; ridge 1; 200 depth-3 XGB rounds; chronological testing; historical features, not refreshed-format claims'},indent=2))
if __name__=='__main__':main()
