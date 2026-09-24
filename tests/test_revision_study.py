import unittest
import numpy as np
from revision_eval.probability import horizon_probability
from revision_eval.study import align_simulation
class StudyTests(unittest.TestCase):
 def baseline(self):return dict(times=[10,20],cum_hazards=[.1,.2],log_shift=0,max_time=20,final_rate=0)
 def test_no_hazard_before_first_event(self):self.assertEqual(float(horizon_probability([0],self.baseline(),5)[0]),0)
 def test_hazard_includes_exact_time(self):self.assertAlmostEqual(float(horizon_probability([0],self.baseline(),10)[0]),1-np.exp(-.1))
 def test_equal_followup_end(self):
  a=dict(initial_wallet={'USD':10},transactions=[dict(timestamp=100)])
  b=dict(initial_wallet={'USD':10},transactions=[dict(timestamp=100),dict(timestamp=700)])
  aa,ha=align_simulation(a,700,604800,.5);bb,hb=align_simulation(b,700,604800,.5)
  self.assertEqual(100+ha,700+hb);self.assertEqual(aa['initial_wallet'],bb['initial_wallet']);self.assertEqual(a['initial_wallet']['USD'],10)
 def test_vector_grid_matches_polynomial(self):
  from revision_eval.trend_grid import polynomial_scores
  grid=np.array([[.6,.4,.1],[.8,.6,.3],[1.,.8,.5]])
  score=lambda w,l,g:(.2+w*.7-l*.3)*(1+g*.4)
  np.testing.assert_allclose(polynomial_scores(score,grid),[score(*p) for p in grid])
 def test_invalid_funding(self):
  with self.assertRaises(ValueError):align_simulation({},700,604800,-1)
if __name__=='__main__':unittest.main()
