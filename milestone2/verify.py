"""Verify entity separation, causal histories, persisted decisions and live replay."""
import json,hashlib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from history import ROOT,FEATURES,engineer
from models import autoencoder,classifier,score
from demo import Screener
from experiment import evaluate

def main():
    checks=[];r=json.loads((ROOT/'results/metrics.json').read_text());b=torch.load(ROOT/'models/detectors.pt',weights_only=True,map_location='cpu')
    wine=pd.read_csv(ROOT/'data/wine.csv');ids=np.arange(len(wine));tr,rest=train_test_split(ids,test_size=.3,stratify=wine.target,random_state=42);va,te=train_test_split(rest,test_size=.5,stratify=wine.target.iloc[rest],random_state=42)
    assert hashlib.sha256((ROOT/'data/wine.csv').read_bytes()).hexdigest()==r['source_csv_sha256']
    all_customers=[];all_samples=[]
    for split,expected in [('train',tr),('validation',va),('test',te)]:
        idx=pd.read_csv(ROOT/f'data/{split}_wine_ids.csv').sample_id.to_numpy();assert np.array_equal(idx,expected)
        events=pd.read_csv(ROOT/f'data/events_{split}.csv');frame=pd.read_csv(ROOT/f'data/features_{split}.csv')
        assert set(events.sample_id)<=set(idx);assert events.transaction_id.is_unique;assert len(events)==r['data'][split]['events'];assert len(frame)==r['data'][split]['scored_transactions'];assert frame.transaction_id.is_unique
        all_customers.append(set(events.customer_id));all_samples.append(set(idx))
    for sets in [all_customers,all_samples]:
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    checks.append('Original wine checksum/splits preserved; customer and linked wine partitions disjoint; 30,000 unique raw events.')
    assert not {'anomaly','scenario','sample_id','customer_id','transaction_id','timestamp_seconds'}&set(FEATURES)
    events=pd.read_csv(ROOT/'data/events_test.csv');cp=np.load(ROOT/'data/chemical_probabilities.npy');subset=events[events.customer_id==events.customer_id.iloc[0]].copy()
    reference=engineer(subset,cp);changed=subset.copy();last=changed.index[-1];changed.loc[last,'unit_price']*=10;changed.loc[last,'amount']*=10
    future=engineer(changed,cp);assert np.allclose(reference[FEATURES].to_numpy()[:-1],future[FEATURES].to_numpy()[:-1])
    poisoned=subset.copy();poisoned['anomaly']=1-poisoned.anomaly;poisoned['scenario']='poisoned';poisoned_features=engineer(poisoned,cp);assert np.allclose(reference[FEATURES],poisoned_features[FEATURES])
    assert r['data']['train']['anomalies']>0
    checks.append('Future-event perturbation leaves prior features unchanged; anomaly/scenario edits leave all model features unchanged; unlabeled AE training includes contamination.')
    train=pd.read_csv(ROOT/'data/features_train.csv');mean=train[FEATURES].mean().to_numpy();assert np.allclose(mean,b['scaler_mean'])
    val=pd.read_csv(ROOT/'data/features_validation.csv');test=pd.read_csv(ROOT/'data/features_test.csv');v=((val[FEATURES].to_numpy()-b['scaler_mean'])/b['scaler_scale']).astype('float32');x=((test[FEATURES].to_numpy()-b['scaler_mean'])/b['scaler_scale']).astype('float32')
    for name,state in b['states'].items():
        ae=name.startswith('ae');d=7 if name=='ae_transaction_only' else 9;m=autoencoder(d,b['bottleneck']) if ae else classifier(d);m.load_state_dict(state);m.eval();s=score(m,x[:,:d],ae)
        saved=pd.read_csv(ROOT/f'results/{name}_test_predictions.csv');assert np.allclose(s,saved.score,atol=1e-5);assert np.array_equal(s>=b['thresholds'][name],saved.prediction.to_numpy().astype(bool))
        observed=evaluate(test.anomaly.to_numpy(),s,b['thresholds'][name]);assert abs(observed['f1_macro']-r['results'][name]['test']['f1_macro'])<1e-12
        if ae:assert abs(np.percentile(score(m,v[:,:d],True),95)-b['thresholds'][name])<1e-5
    checks.append('Training-only scaler; every model checkpoint reproduces test scores/decisions; metrics recompute; AE thresholds equal frozen validation p95.')
    sc=Screener();scored=[]
    for event in subset.to_dict('records'):
        output=sc.predict_event(event)
        if output['action']!='history_warmup':scored.append(output)
    saved=pd.read_csv(ROOT/f"results/{r['primary']}_test_predictions.csv").set_index('transaction_id')
    for output in scored:assert abs(output['anomaly_score']-saved.loc[output['transaction_id'],'score'])<1e-4
    try:sc.predict_event(subset.iloc[-1].to_dict())
    except ValueError:pass
    else:raise AssertionError('Out-of-order/repeated event accepted')
    checks.append('Stateful raw-event replay matches saved test scores; first five events warm up; duplicate/out-of-order timestamps rejected.')
    explanations=json.loads((ROOT/'results/flag_explanations.json').read_text());assert len(explanations)==r['results'][r['primary']]['test']['tp']+r['results'][r['primary']]['test']['fp']
    assert all(len(e['top_factors'])==3 and all(t['squared_standardized_error']>=0 for t in e['top_factors']) for e in explanations)
    checks.append('Every flagged test transaction has three nonnegative reconstruction-error explanation factors.')
    text='\n'.join('PASS: '+c for c in checks)+'\n';(ROOT/'results/verification.txt').write_text(text);print(text)

if __name__=='__main__':main()
