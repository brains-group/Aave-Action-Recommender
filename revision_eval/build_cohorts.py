"""Freeze disjoint stress and outcome-independent user samples from held-out time."""
import argparse,hashlib,json,pickle
from pathlib import Path
import pandas as pd

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def key(row):return ':'.join(str(row[k]) for k in ['user','timestamp','pool','Index Event'])
def rank(s):return hashlib.sha256(('revision-42:'+str(s)).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--profiles',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--size',type=int,default=200);p.add_argument('--test-start',type=int,required=True);p.add_argument('--end',type=int,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 eligible={f.stem[5:] for f in a.profiles.glob('*/profiles/user_*.json')}
 sources=sorted(a.data.glob('*/Liquidated/data.csv'));stress={};normal={};identity=[]
 # Choose the normal sample independently first. Keep 2*n stress candidates
 # so removing up to n normal accounts still leaves n stress accounts.
 for source in sources:
  st=source.stat();identity.append(dict(path=str(source.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha(source)))
  for chunk in pd.read_csv(source,chunksize=100000):
   chunk=chunk[(chunk.timestamp>=a.test_start)&(chunk.timestamp+600+604800<=a.end)&chunk.user.isin(eligible)]
   for row in chunk.to_dict('records'):
    user=row['user'];row['case_id']=key(row)
    score=(rank(user),rank(row['case_id']))
    if user not in normal or score<normal[user][0]:normal[user]=(score,row)
    if row['status']==1 and row['timeDiff']>=0:
     score=(float(row['timeDiff']),rank(row['case_id']))
     if user not in stress or score<stress[user][0]:stress[user]=(score,row)
   normal=dict(sorted(normal.items(),key=lambda x:x[1][0])[:a.size])
   stress=dict(sorted(stress.items(),key=lambda x:x[1][0])[:2*a.size])
  print('scanned',source.parent.parent.name,'normal',len(normal),'stress candidates',len(stress),flush=True)
 stress=dict(sorted(((u,r) for u,r in stress.items() if u not in normal),key=lambda x:x[1][0])[:a.size])
 for name,items in [('high_risk',stress),('normal',normal)]:
  rows=[x[1] for x in sorted(items.values(),key=lambda x:x[0])];pd.DataFrame(rows).to_csv(a.output/(name+'.csv'),index=False)
 assert not set(stress)&set(normal)
 (a.output/'manifest.json').write_text(json.dumps(dict(size_per_cohort=a.size,test_start=a.test_start,end=a.end,sources=identity,normal='hash sample of profile-covered eligible users, then hash checkpoint; future labels unused',high_risk='shortest observed time-to-liquidation per selected user in held-out window',eligibility='existing profile, feature-bearing core checkpoint, seven-day follow-up availability by dataset end; no zero-debt exclusions',sampling_seed='revision-42',account_overlap=0,selection_order='normal first independent of outcomes; stress excludes already selected normal users',model_split='chronological; not a claim of unseen training accounts',original_12000_preserved=True),indent=2))
if __name__=='__main__':main()
