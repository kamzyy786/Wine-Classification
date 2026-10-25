"""Load the measured model and screen a synthetic purchase. No retraining."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import torch
from experiment import network, chemical_features, probs, TX, ROOT

class Screener:
    def __init__(self):
        b=torch.load(ROOT/'models/screening.pt',map_location='cpu',weights_only=True)
        self.b=b
        self.chem=network(13,b['chemical_width'],3); self.chem.load_state_dict(b['chem_state']); self.chem.eval()
        self.risk=network(len(b['risk_columns']),b['risk_width'],2); self.risk.load_state_dict(b['risk_state']); self.risk.eval()

    def predict(self,measurements,declared,transaction):
        raw=np.asarray(measurements,dtype=float)
        if raw.shape!=(13,) or not np.isfinite(raw).all() or (raw<=0).any():
            raise ValueError('Provide 13 finite positive chemical measurements in source feature order.')
        if declared not in [0,1,2]: raise ValueError('Declared cultivar must be 0, 1, or 2.')
        tx=np.asarray(transaction,dtype=float)
        if tx.shape!=(4,) or not np.isfinite(tx).all() or (tx<0).any() or tx[3] not in [0,1]:
            raise ValueError('Provide nonnegative price, quantity, recent purchases, and a 0/1 new-category indicator.')
        b=self.b; z=(raw-np.array(b['chemical_mean']))/np.array(b['chemical_scale'])
        p=probs(self.chem,z[None,:]); chem=chemical_features(p,np.array([declared]))
        ff=np.concatenate([tx,chem[0]])[b['risk_columns']]
        rz=(ff-np.array(b['risk_mean']))/np.array(b['risk_scale'])
        score=float(probs(self.risk,rz[None,:])[0,1]); reasons=[]
        contributions=[]
        groups=[([i],label) for i,label in enumerate(TX)]+[([4,5],'chemical_consistency')]
        for original,label in groups:
            positions=[b['risk_columns'].index(i) for i in original if i in b['risk_columns']]
            if not positions: continue
            occluded=rz.copy(); occluded[positions]=0
            delta=score-float(probs(self.risk,occluded[None,:])[0,1])
            contributions.append({'factor':label,'risk_score_drop_at_training_mean':delta})
        contributions.sort(key=lambda a:a['risk_score_drop_at_training_mean'],reverse=True)
        if int(p.argmax())!=declared: reasons.append('Predicted cultivar differs from declared cultivar; investigate laboratory/label consistency.')
        # Descriptive standardized deviations are not causal feature attributions.
        for i,label in enumerate(TX):
            if abs((tx[i]-b['risk_mean'][i])/b['risk_scale'][i])>=2:
                reasons.append(f'{label} differs by at least two training standard deviations.')
        if not reasons: reasons.append('No simple diagnostic rule triggered; review the model score and supporting record.')
        return {'synthetic_demo':True,'risk_score':score,'threshold':b['threshold'],'action':'human_review' if score>=b['threshold'] else 'no_model_flag','predicted_cultivar':int(p.argmax()),'declared_cultivar':declared,'chemical_disagreement_probability':float(chem[0,0]),'top_factors':contributions[:3],'diagnostic_factors':reasons,'interpretation':'Grouped mean-occlusion diagnostics, not causal explanations. Screening signal, not proof of fraud or chemical safety.'}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--row',type=int,default=0); args=parser.parse_args()
    tx=pd.read_csv(ROOT/'data/synthetic_test.csv').iloc[args.row]
    wine=pd.read_csv(ROOT/'data/wine.csv').drop(columns='target').iloc[int(tx.sample_id)].to_numpy()
    print(json.dumps({'transaction_id':tx.transaction_id,**Screener().predict(wine,int(tx.declared_cultivar),tx[TX].to_numpy())},indent=2))

if __name__=='__main__': main()
