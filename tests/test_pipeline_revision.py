import ast
from pathlib import Path
import unittest
import logging
import numpy as np

class RepaymentTests(unittest.TestCase):
    def helper(self):
        p=Path(__file__).parents[1]/'utils/simulations.py'
        tree=ast.parse(p.read_text())
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'updateAmountOrUSD','update_recommendation_if_necessary'}]
        ns={'np':np,'logger':logging.getLogger(__name__)}
        exec(compile(ast.Module(body=funcs,type_ignores=[]),str(p),'exec'),ns)
        return ns['update_recommendation_if_necessary']

    def test_same_asset_only_no_upsizing(self):
        fn=self.helper()
        rec={'Index Event':'repay','reserve':'USD','timestamp':20,'amount':10,'priceInUSD':1}
        result={'checkpoint_state':{'timestamp':10,'wallet_balances':{'USD':5,'ETH':100},'debt_balances':{'USD':8}}}
        new=fn(rec,result)
        self.assertEqual(new['amount'],5)
        self.assertEqual(new['reserve'],'USD')
        self.assertEqual(rec['amount'],10)
        result['checkpoint_state']['timestamp']=30
        self.assertIsNone(fn(rec,result))
        self.assertIsNone(fn(rec,{'final_state':{}}))
