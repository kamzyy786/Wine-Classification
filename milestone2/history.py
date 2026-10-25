"""Raw-event simulation and strictly past-only customer feature engineering."""
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BEHAVIOUR=['amount_deviation','quantity_deviation','log_purchases_last_hour','log_minutes_since_previous','new_merchant_category','log_location_shift_km','log_unit_price']
CHEMISTRY=['chemical_disagreement','cultivar_mismatch']
FEATURES=BEHAVIOUR+CHEMISTRY
SCENARIOS=['amount','velocity','merchant','location','chemical_mismatch','combined']

def distance_km(a,b):
    la1,lo1,la2,lo2=np.radians([*a,*b]);dlat=la2-la1;dlon=lo2-lo1
    return float(6371*2*np.arcsin(np.sqrt(np.clip(np.sin(dlat/2)**2+np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2,0,1))))

def generate_events(sample_ids,true_cultivar,n_customers,seed,nominal_anomaly_probability,split):
    rng=np.random.default_rng(seed);records=[]
    for customer in range(n_customers):
        cid=f'{split}-C{customer:04d}';clock=float(rng.uniform(0,86400*7));home=(float(rng.uniform(43,49)),float(rng.uniform(-76,-69)))
        base_price=float(rng.lognormal(np.log(25),.25));base_quantity=int(rng.integers(2,6));base_gap=float(rng.uniform(3,20)*3600)
        usual_categories=rng.choice(8,3,replace=False);burst=0
        for j in range(30):
            if j<5:kind='normal'
            elif burst>0:kind='velocity';burst-=1
            elif rng.random()<nominal_anomaly_probability:kind=str(rng.choice(SCENARIOS,p=[.2,.2,.15,.15,.2,.1]))
            else:kind='normal'
            bad_amount=kind in ['amount','combined'];bad_velocity=kind in ['velocity','combined'];bad_chemical=kind in ['chemical_mismatch','combined']
            if bad_velocity:
                gap=float(rng.uniform(3,15))
                if burst==0 and kind!='velocity':burst=2
                elif burst==0 and j>5 and records and records[-1]['scenario']!='velocity':burst=2
            else:gap=float(max(120,rng.exponential(base_gap)))
            clock+=gap
            price=float(rng.lognormal(np.log(base_price),.15));qty=max(1,int(rng.poisson(base_quantity)))
            if bad_amount:
                price*=float(rng.choice([.12,7.5]));qty+=int(rng.integers(12,30))
            merchant=int(rng.choice(usual_categories))
            if kind in ['merchant','combined']:merchant=100+j
            # Rare legitimate novelty and travel prevent a perfectly deterministic signal.
            elif rng.random()<.003:merchant=200+j
            loc=(home[0]+float(rng.normal(0,.015)),home[1]+float(rng.normal(0,.015)))
            if kind in ['location','combined'] or rng.random()<.003:loc=(home[0]+float(rng.uniform(3,7)),home[1]+float(rng.uniform(3,7)))
            sid=int(rng.choice(sample_ids));declared=int(true_cultivar[sid])
            if rng.random()<(.97 if bad_chemical else .003):declared=int((declared+rng.integers(1,3))%3)
            records.append({'transaction_id':f'{cid}-T{j:02d}','customer_id':cid,'timestamp_seconds':clock,'unit_price':price,'quantity':qty,'amount':price*qty,'merchant_category':merchant,'latitude':loc[0],'longitude':loc[1],'sample_id':sid,'declared_cultivar':declared,'anomaly':int(kind!='normal'),'scenario':kind,'event_index':j})
    return pd.DataFrame(records)

class CustomerHistory:
    """State updates only AFTER deriving a current event's features."""
    def __init__(self):
        self.amounts=[];self.quantities=[];self.times=deque();self.categories=set();self.previous_time=None;self.previous_location=None

    def transform_then_update(self,event):
        clock=float(event['timestamp_seconds']);amount=float(event['amount']);quantity=float(event['quantity']);price=float(event['unit_price'])
        if amount<=0 or quantity<=0 or price<=0:raise ValueError('Amounts, quantity and price must be positive')
        if abs(amount-price*quantity)>1e-5*max(1,amount):raise ValueError('Amount must equal price times quantity')
        if self.previous_time is not None and clock<=self.previous_time:raise ValueError('Events must have strictly increasing customer timestamps')
        while self.times and self.times[0]<clock-3600:self.times.popleft()
        la=np.log(amount);lq=np.log1p(quantity);loc=(float(event['latitude']),float(event['longitude']))
        features=[(la-np.mean(self.amounts))/max(float(np.std(self.amounts)),.25) if self.amounts else 0.,(lq-np.mean(self.quantities))/max(float(np.std(self.quantities)),.2) if self.quantities else 0.,float(np.log1p(len(self.times))),float(np.log1p((clock-self.previous_time)/60)) if self.previous_time is not None else 0.,float(event['merchant_category'] not in self.categories),float(np.log1p(distance_km(self.previous_location,loc))) if self.previous_location else 0.,float(np.log(price))]
        features[0]=float(np.clip(features[0],-15,15));features[1]=float(np.clip(features[1],-15,15))
        self.amounts.append(la);self.quantities.append(lq);self.times.append(clock);self.categories.add(event['merchant_category']);self.previous_time=clock;self.previous_location=loc
        return features

def engineer(events,chemical_probabilities):
    rows=[]
    for _,customer in events.groupby('customer_id',sort=False):
        state=CustomerHistory()
        for event in customer.sort_values('timestamp_seconds').to_dict('records'):
            beh=state.transform_then_update(event);sid=int(event['sample_id']);declared=int(event['declared_cultivar']);p=chemical_probabilities[sid]
            if event['event_index']<5:continue
            row={k:event[k] for k in ['transaction_id','customer_id','sample_id','timestamp_seconds','anomaly','scenario']}
            row.update(dict(zip(FEATURES,beh+[float(1-p[declared]),float(p.argmax()!=declared)])));rows.append(row)
    return pd.DataFrame(rows)
