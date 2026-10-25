"""Add pre-specified supervised baselines to an existing frozen AE experiment."""
import json,time,hashlib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from history import ROOT,FEATURES,SCENARIOS
from models import build_model,input_width,score,hardware,rss_mib
from experiment import train_model,evaluate

def main():
    start=time.perf_counter();r=json.loads((ROOT/'results/metrics.json').read_text());path=ROOT/'models/detectors.pt';b=torch.load(path,weights_only=True,map_location='cpu')
    names=['mlp_transaction_only','linear_combined']
    if any(n in b['states'] for n in names):raise RuntimeError('Baselines already exist. Run experiment.py for a fresh full reproduction.')
    old_primary={k:v.clone() for k,v in b['states'][b['primary']].items()};threshold=b['thresholds'][b['primary']];frozen_bytes=(ROOT/'results/frozen_ae_selection.json').read_bytes()
    frames={k:pd.read_csv(ROOT/f'data/features_{k}.csv') for k in ['train','validation','test']};scaled={k:((v[FEATURES].to_numpy()-np.array(b['scaler_mean']))/np.array(b['scaler_scale'])).astype('float32') for k,v in frames.items()}
    for name in names:
        torch.manual_seed(42);m=build_model(name);d=input_width(name)
        log=train_model(m,scaled['train'][:,:d],scaled['validation'][:,:d],frames['train'].anomaly.to_numpy(),frames['validation'].anomaly.to_numpy(),weighted=True)
        vs=score(m,scaled['validation'][:,:d],False);t=float(max(np.linspace(.05,.95,37),key=lambda q:f1_score(frames['validation'].anomaly,vs>=q,average='macro')))
        ts=score(m,scaled['test'][:,:d],False);pred=ts>=t
        result={'validation':evaluate(frames['validation'].anomaly.to_numpy(),vs,t),'test':evaluate(frames['test'].anomaly.to_numpy(),ts,t),'test_scenario_recall':{s:float(pred[frames['test'].scenario==s].mean()) for s in SCENARIOS}}
        pd.DataFrame({'transaction_id':frames['test'].transaction_id,'customer_id':frames['test'].customer_id,'sample_id':frames['test'].sample_id,'scenario':frames['test'].scenario,'actual':frames['test'].anomaly,'score':ts,'prediction':pred.astype(int)}).to_csv(ROOT/f'results/{name}_test_predictions.csv',index=False)
        hist=pd.DataFrame(log.pop('history'),columns=['epoch','train_loss','validation_loss','validation_metric']);hist.to_csv(ROOT/f'results/{name}_learning.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(9,3));axes[0].plot(hist.epoch,hist.train_loss,label='Train');axes[0].plot(hist.epoch,hist.validation_loss,label='Validation');axes[0].legend();axes[0].set(xlabel='Epoch',ylabel='Cross entropy');axes[1].plot(hist.epoch,hist.validation_metric);axes[1].set(xlabel='Epoch',ylabel='Validation macro F1');fig.tight_layout();fig.savefig(ROOT/f'results/{name}_learning.png',dpi=150);plt.close(fig)
        b['states'][name]=m.state_dict();b['thresholds'][name]=t;r['thresholds'][name]=t;r['results'][name]=result;r['training'][name]=log
        print(name,result['test'],flush=True)
    assert b['thresholds'][b['primary']]==threshold
    assert all(torch.equal(v,b['states'][b['primary']][k]) for k,v in old_primary.items())
    assert (ROOT/'results/frozen_ae_selection.json').read_bytes()==frozen_bytes
    torch.save(b,path);r['checkpoint_bytes']=path.stat().st_size
    r['supplementary_comparisons']={'models':names,'seconds':time.perf_counter()-start,'environment':hardware(),'process_rss_after_mib':rss_mib(),'primary_ae_changed':False,'method':'Added fixed 7-16-2 transaction MLP and 9-2 linear supervised baselines to the same existing partitions; validation thresholds selected before test scoring; no AE or data refit.'}
    (ROOT/'results/metrics.json').write_text(json.dumps(r,indent=2));state=json.loads((ROOT/'PROJECT_STATE.json').read_text());state['detectors_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();state['audit_revision']='3.1 complete supervised comparisons';(ROOT/'PROJECT_STATE.json').write_text(json.dumps(state,indent=2))

if __name__=='__main__':main()
