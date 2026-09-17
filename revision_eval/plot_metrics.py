"""Readable standalone figures from new predictive benchmark outputs."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
x=json.loads(a.results.read_text());tasks=[k for k,v in x.items() if v['models']];models=['KM','CoxPH','XGBoostCox'];fig,axes=plt.subplots(len(tasks),2,figsize=(10,3*len(tasks)),squeeze=False,layout='constrained')
for i,task in enumerate(tasks):
 for model in models:
  r=x[task]['models'].get(model,{})
  if 'known_label_7d' not in r:continue
  c=r['known_label_7d'];bins=[b for b in c['calibration'] if b['n']];axes[i,0].plot([b['mean_prediction'] for b in bins],[b['event_fraction'] for b in bins],'o-',label=model)
  auc=r.get('dynamic_auc')
  if isinstance(auc,list):axes[i,1].plot([1,3,7],auc,'o-',label=model)
 axes[i,0].plot([0,1],[0,1],'k--',alpha=.4);axes[i,0].set(title=task+' → Liquidation: calibration',xlabel='Mean predicted probability (7 days)',ylabel='Observed fraction (known-label subset)')
 axes[i,1].set(title=task+' → Liquidation: IPCW AUC',xlabel='Horizon (days)',ylabel='Cumulative/dynamic AUC',ylim=(0,1));axes[i,1].legend()
fig.savefig(a.output/'predictive-diagnostics.pdf');plt.close(fig)
