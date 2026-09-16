import unittest
from revision_eval.policies import propose
from revision_eval.sensitivity import first_flip
from revision_eval.metrics import horizon_metrics
from revision_eval.cohorts import representative
class RevisionTests(unittest.TestCase):
 def state(self):return dict(case_id='a',timestamp=100,state_timestamp=100,total_debt_usd=100,weighted_collateral_usd=105,assets={'USD':dict(price_usd=1,wallet=20,debt=100,liquidation_threshold=.8,collateral_enabled=True)})
 def test_repay_budget(self):
  r=propose(self.state(),'repay_only',budget_usd=10);self.assertEqual(r['amount'],10);self.assertFalse(r['target_reached'])
 def test_hf_target(self):
  r=propose(self.state(),'static_hf');self.assertEqual(r['action'],'Repay');self.assertAlmostEqual(r['amount'],12.5)
 def test_deposit(self):self.assertAlmostEqual(propose(self.state(),'deposit_only')['amount'],18.75)
 def test_no_conversion(self):
  s=self.state();s['assets']['USD']['wallet']=0;self.assertIsNone(propose(s,'repay_only')['action'])
 def test_future(self):
  s=self.state();s['state_timestamp']=101
  with self.assertRaises(ValueError):propose(s,'static_hf')
 def test_nonmonotonic(self):
  g=[dict(combinations=[(x,0,0)]) for x in [.1,.2,.3]]
  self.assertAlmostEqual(first_flip(lambda c:c[0]==.1,(0,0,0),g)['min_distance'],.1)
 def test_error_not_stable(self):
  def f(c):
   if c[0]:raise ValueError('missing model')
   return False
  with self.assertRaises(ValueError):first_flip(f,(0,),[dict(combinations=[(.1,)])])
 def test_censoring_and_ties(self):
  rows=[dict(case_id=str(i),duration=t,event=e,probability=p,horizon=10) for i,(t,e,p) in enumerate([(10,1,.8),(10,0,.2),(9,0,.9),(11,1,.1)])]
  m=horizon_metrics(rows,10);self.assertEqual((m['tp'],m['tn'],m['n']),(1,2,3));self.assertEqual(m['excluded_censored_ids'],['2'])
 def test_cohort_ignores_outcomes(self):
  r=[dict(case_id=str(i),account=str(i),split='test',eligible=True,event=i%2) for i in range(10)]
  a=representative(r,3);b=representative(list(reversed(r)),3);self.assertEqual(a,b)
if __name__=='__main__':unittest.main()
