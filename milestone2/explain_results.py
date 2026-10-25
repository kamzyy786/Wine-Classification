"""Explain every test flag and estimate sample-cluster uncertainty."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from demo import Screener
from experiment import ROOT, TX

def main():
    tx=pd.read_csv(ROOT/'data/synthetic_test.csv'); wines=pd.read_csv(ROOT/'data/wine.csv').drop(columns='target')
    s=Screener(); results=[]
    for _,row in tx.iterrows():
        d=s.predict(wines.iloc[int(row.sample_id)].to_numpy(),int(row.declared_cultivar),row[TX].to_numpy())
        results.append({'transaction_id':row.transaction_id,'actual_scenario':row.scenario,**d})
    (ROOT/'results/test_explanations.json').write_text(json.dumps(results,indent=2))
    r=json.loads((ROOT/'results/metrics.json').read_text()); selected=r['selected']
    full=pd.read_csv(ROOT/f'results/{selected}_test_predictions.csv'); base=pd.read_csv(ROOT/'results/transaction_only_test_predictions.csv')
    rng=np.random.default_rng(2026); groups=tx.sample_id.unique(); diffs=[]
    for _ in range(1000):
        sampled=rng.choice(groups,len(groups),replace=True); idx=np.concatenate([np.where(tx.sample_id.to_numpy()==g)[0] for g in sampled])
        diffs.append(f1_score(full.actual.iloc[idx],full.prediction.iloc[idx],average='macro')-f1_score(base.actual.iloc[idx],base.prediction.iloc[idx],average='macro'))
    summary={'method':'Paired bootstrap over 27 held-out wine sample clusters; 1000 replicates; simulator fixed. Exploratory interval, not external validation.','macro_f1_difference':float(f1_score(full.actual,full.prediction,average='macro')-f1_score(base.actual,base.prediction,average='macro')),'percentile_95_interval':np.percentile(diffs,[2.5,97.5]).tolist(),'explained_flags':sum(d['action']=='human_review' for d in results),'all_predictions_explained':len(results)}
    (ROOT/'results/contribution_analysis.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
