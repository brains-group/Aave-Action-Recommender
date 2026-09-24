import unittest
from utils.recommendation_funding import funding_choice, NoFeasibleAction

class FundingTests(unittest.TestCase):
    def test_repay_requires_same_asset(self):
        state={'timestamp':100,'wallet_balances':{'ETH':100,'USDC':20},'debt_balances':{'USDC':12}}
        self.assertEqual(funding_choice(state,'Repay',100,lambda s:2000 if s=='ETH' else 1),('USDC',12,12))
    def test_minimum_is_usd(self):
        state={'timestamp':100,'wallet_balances':{'ETH':.1},'debt_balances':{}}
        self.assertEqual(funding_choice(state,'Deposit',100,lambda s:2000),('ETH',.025,.1))
    def test_abstain_without_repay_asset(self):
        with self.assertRaises(NoFeasibleAction):
            funding_choice({'timestamp':100,'wallet_balances':{'ETH':10},'debt_balances':{'USDC':100}},'Repay',100,lambda s:1)
    def test_future_checkpoint_rejected(self):
        with self.assertRaises(NoFeasibleAction):
            funding_choice({'timestamp':101,'wallet_balances':{'ETH':10}},'Deposit',100,lambda s:1)
    def test_nonfinite_price_rejected(self):
        with self.assertRaises(NoFeasibleAction):
            funding_choice({'timestamp':100,'wallet_balances':{'ETH':10}},'Deposit',100,lambda s:float('nan'))
if __name__=='__main__':unittest.main()
