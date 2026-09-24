import os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        os.environ['AAVE_EVALUATION_CACHE']=cls.tmp.name
        os.environ['AAVE_RECOMMENDATIONS_FILE']='/home/spadef/Aave-Action-Recommender/cache/recommendations.pkl'
        import actionAgentTraining, performSimulations
        cls.agent=actionAgentTraining; cls.sim=performSimulations
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_abstention_has_identical_arms(self):
        p=self.sim; profile={'user_address':'test','transactions':[{'timestamp':100,'action':'Deposit'}]}
        rec={'user':'test','timestamp':700,'Index Event':'deposit','reserve':'USDC','amount':0,'abstained':True};calls=[]
        def simulate(rec,suffix,**kw):
            calls.append(kw)
            return {'liquidation_stats':{'liquidated':True,'liquidation_reason':'test'},'final_state':{'total_debt_usd':0}}
        p.outputFile=str(Path(self.tmp.name)/'strategy.log')
        with patch.object(p,'get_limited_user_profile',return_value=(profile,700,100,'test',[],86400)),patch.object(p,'get_simulation_outcome',side_effect=simulate),patch.object(p.UserProfileGenerator,'_row_to_transaction',side_effect=AssertionError('Abstention must not become a transaction')):
            p.process_recommendation((rec,{}))
        self.assertEqual(len(calls),2);self.assertEqual(calls[0]['profile'],calls[1]['profile']);self.assertEqual(calls[0]['lookahead_seconds'],calls[1]['lookahead_seconds'])
    def test_optimizer_ignores_projected_wallet(self):
        a=self.agent;row=pd.Series({'user':'test','timestamp':100})
        def next_action(row,action,amount=10,reserve=None):return pd.Series({'amount':amount,'reserve':reserve,'timestamp':700})
        result={'checkpoint_state':{'timestamp':100,'wallet_balances':{'USDC':20,'ETH':500},'debt_balances':{'USDC':12}},'final_state':{'wallet_balances':{'ETH':99999}}}
        with patch.object(a,'generate_next_transaction',side_effect=next_action),patch.object(a,'get_limited_user_profile',return_value=({},None,None,None,None,100)),patch.object(a,'get_simulation_outcome',return_value=result),patch.object(a,'get_price_history_value',return_value=1),patch.object(a,'determine_liquidation_risk',return_value=(False,)):
            action=a.optimize_recommendation(row,'Repay')
        self.assertEqual(action['reserve'],'USDC');self.assertEqual(action['amount'],12)
    def test_prediction_cache_distinguishes_action(self):
        a=self.agent;base=pd.Series({'user':'test','timestamp':100,'amount':10,'Index Event':'deposit','reserve':'USDC'});calls=[]
        def calc(event,group,results,date,history):calls.append(event);results[100]={'event':event}
        dates=pd.DatetimeIndex([pd.Timestamp('1970-01-01')])
        with patch.object(a,'RESULTS_CACHE_DIR',self.tmp.name),patch.object(a,'get_date_ranges',return_value=(dates,dates)),patch.object(a,'get_user_history',return_value=pd.DataFrame()),patch.object(a,'calc_predictions',side_effect=calc):
            one=a.get_transaction_history_predictions(base);other=base.copy();other['Index Event']='repay';two=a.get_transaction_history_predictions(other);again=a.get_transaction_history_predictions(base)
        self.assertNotEqual(one,two);self.assertEqual(one,again);self.assertEqual(len(calls),2)
if __name__=='__main__':unittest.main()
