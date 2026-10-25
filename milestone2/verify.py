"""Checks actual separation, preserved source split, inference and saved metrics."""
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from demo import Screener
from experiment import ROOT, TX

def main():
    r=json.loads((ROOT/'results/metrics.json').read_text()); wine=pd.read_csv(ROOT/'data/wine.csv')
    assert hashlib.sha256((ROOT/'data/wine.csv').read_bytes()).hexdigest()==r['source_sha256']
    ids=np.arange(178); tr,rest=train_test_split(ids,test_size=.3,stratify=wine.target,random_state=42); va,te=train_test_split(rest,test_size=.5,stratify=wine.target.iloc[rest],random_state=42)
    sets=[]
    for name,expected in [('train',tr),('validation',va),('test',te)]:
        actual=pd.read_csv(ROOT/f'results/{name}_sample_ids.csv').sample_id.to_numpy(); assert np.array_equal(actual,expected)
        tx=pd.read_csv(ROOT/f'data/synthetic_{name}.csv'); assert set(tx.sample_id)==set(expected); assert tx.transaction_id.is_unique; sets.append(set(tx.sample_id))
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    s=Screener(); tx=pd.read_csv(ROOT/'data/synthetic_test.csv'); pred=pd.read_csv(ROOT/f"results/{r['selected']}_test_predictions.csv")
    for i in [0,1,20,100,539]:
        row=tx.iloc[i]; result=s.predict(wine.drop(columns='target').iloc[int(row.sample_id)].to_numpy(),int(row.declared_cultivar),row[TX].to_numpy())
        assert abs(result['risk_score']-pred.probability.iloc[i])<1e-6
        assert len(result['top_factors'])==3
    bad_inputs=[([1]*12,0,[25,1,0,0]),([float('nan')]*13,0,[25,1,0,0]),([1]*13,3,[25,1,0,0]),([1]*13,0,[-1,1,0,0])]
    for args in bad_inputs:
        try: s.predict(*args)
        except ValueError: pass
        else: raise AssertionError('Invalid input accepted')
    for c in r['risk_candidates']:
        p=pd.read_csv(ROOT/f"results/{c['name']}_test_predictions.csv")
        assert abs(accuracy_score(p.actual,p.prediction)-c['test']['accuracy'])<1e-12
        assert abs(f1_score(p.actual,p.prediction,average='macro')-c['test']['f1_macro'])<1e-12
    print('PASS: source checksum; original splits; group separation; unique transactions; checkpoint-score agreement; input rejection; all saved test accuracy/F1 metrics.')
    (ROOT/'results/verification.txt').write_text('PASS: source checksum, original splits, group separation, unique transactions, checkpoint-score agreement, input rejection, saved metrics.\n')

if __name__=='__main__': main()
