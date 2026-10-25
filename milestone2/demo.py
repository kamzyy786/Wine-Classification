"""Stateful saved-model demonstration: raw events -> past-only features -> AE flag."""
import argparse,json
import numpy as np
import pandas as pd
import torch
from chemistry import network,probs
from models import autoencoder,classifier,score
from history import ROOT,FEATURES,CustomerHistory
torch.set_num_threads(1)

class Screener:
    def __init__(self):
        self.bundle=torch.load(ROOT/'models/detectors.pt',weights_only=True,map_location='cpu');b=self.bundle
        self.model=autoencoder(9,b['bottleneck']);self.model.load_state_dict(b['states'][b['primary']]);self.model.eval()
        self.mean=np.array(b['scaler_mean']);self.scale=np.array(b['scaler_scale']);self.threshold=b['thresholds'][b['primary']];self.histories={}
        c=torch.load(ROOT/'models/chemistry.pt',weights_only=True,map_location='cpu');self.chemistry=c
        self.chem=network(13,c['chemical_width'],3);self.chem.load_state_dict(c['chem_state']);self.chem.eval()
        self.wines=pd.read_csv(ROOT/'data/wine.csv').drop(columns='target').to_numpy()

    def predict_features(self,features):
        x=np.asarray([features[f] for f in FEATURES],dtype=float)
        if not np.isfinite(x).all():raise ValueError('All features must be finite')
        z=((x-self.mean)/self.scale).astype('float32')
        with torch.inference_mode():reconstruction=self.model(torch.tensor(z)[None,:]).numpy()[0]
        errors=(z-reconstruction)**2;s=float(errors.mean());top=np.argsort(errors)[::-1][:3]
        return {'anomaly_score':s,'threshold':self.threshold,'action':'human_review' if s>=self.threshold else 'no_model_flag','top_factors':[{'feature':FEATURES[j],'squared_standardized_error':float(errors[j]),'share_of_total_error':float(errors[j]/max(errors.sum(),1e-12))} for j in top],'interpretation':'Uncalibrated reconstruction error; not a fraud probability or proof of wrongdoing.'}

    def predict_event(self,event):
        sid=int(event['sample_id']);declared=int(event['declared_cultivar'])
        if sid<0 or sid>=len(self.wines) or declared not in [0,1,2]:raise ValueError('Invalid sample or declared cultivar')
        if not np.isfinite(float(event['timestamp_seconds'])):raise ValueError('Timestamp must be finite')
        for f in ['unit_price','quantity','amount','latitude','longitude']:
            if not np.isfinite(float(event[f])):raise ValueError('Event values must be finite')
        if not -90<=float(event['latitude'])<=90 or not -180<=float(event['longitude'])<=180:raise ValueError('Invalid coordinates')
        state=self.histories.setdefault(str(event['customer_id']),CustomerHistory());behaviour=state.transform_then_update(event)
        if len(state.amounts)<=5:return {'transaction_id':event['transaction_id'],'action':'history_warmup','history_events':len(state.amounts)}
        c=self.chemistry;z=(self.wines[sid]-np.array(c['chemical_mean']))/np.array(c['chemical_scale']);p=probs(self.chem,z[None,:])[0]
        values=dict(zip(FEATURES,behaviour+[float(1-p[declared]),float(p.argmax()!=declared)]))
        return {'transaction_id':event['transaction_id'],'predicted_cultivar':int(p.argmax()),'declared_cultivar':declared,'derived_features':values,**self.predict_features(values)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--customer',default='test-C0000');parser.add_argument('--event',type=int,default=None);args=parser.parse_args()
    events=pd.read_csv(ROOT/'data/events_test.csv');events=events[events.customer_id==args.customer].sort_values('timestamp_seconds')
    if events.empty:raise ValueError('Customer not in test events')
    screen=Screener()
    for event in events.to_dict('records'):
        if args.event is not None and event['event_index']>args.event:break
        result=screen.predict_event(event)
        if args.event is None or event['event_index']==args.event:print(json.dumps(result))

if __name__=='__main__':main()
