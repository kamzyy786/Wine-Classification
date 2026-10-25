"""Measure a separate inference process; no training runs in this script."""
import json,time
import numpy as np
import pandas as pd
import torch
from history import ROOT,FEATURES
from models import autoencoder,classifier,score,rss_mib,hardware,dense_flops
torch.set_num_threads(1)

def main():
    baseline=rss_mib();b=torch.load(ROOT/'models/detectors.pt',weights_only=True,map_location='cpu');mean=np.array(b['scaler_mean']);scale=np.array(b['scaler_scale']);raw=pd.read_csv(ROOT/'data/features_test.csv')[FEATURES].to_numpy();measurements={}
    for name,state in b['states'].items():
        ae=name.startswith('ae');d=7 if name=='ae_transaction_only' else 9;model=autoencoder(d,b['bottleneck']) if ae else classifier(d);model.load_state_dict(state);model.eval();runs=[]
        for batch in [1,32]:
            x=raw[:batch,:d]
            def fn():return score(model,((x-mean[:d])/scale[:d]).astype('float32'),ae)
            for _ in range(100):fn()
            times=[]
            for _ in range(1000):
                t=time.perf_counter_ns();fn();times.append((time.perf_counter_ns()-t)/1e6)
            runs.append({'batch':batch,'p50_ms':float(np.percentile(times,50)),'p90_ms':float(np.percentile(times,90)),'p95_ms':float(np.percentile(times,95)),'samples_per_second':float(batch*1000/np.mean(times))})
        measurements[name]={'latency':runs,'process_rss_mib':rss_mib(),'parameters':sum(p.numel() for p in model.parameters()),'dense_forward_flops':dense_flops(model)}
    # Full event path: rebuild the fixed prefix outside each timed call so all calls
    # see the same history; only the current event's feature/chemistry/AE work is timed.
    from demo import Screener
    from history import CustomerHistory
    import copy
    events=pd.read_csv(ROOT/'data/events_test.csv');customer=events.customer_id.iloc[0];stream=events[events.customer_id==customer].sort_values('timestamp_seconds').to_dict('records');s=Screener()
    for event in stream[:10]:s.predict_event(event)
    prefix=copy.deepcopy(s.histories[customer]);event=stream[10];times=[]
    for i in range(1100):
        s.histories[customer]=copy.deepcopy(prefix)
        t=time.perf_counter_ns();s.predict_event(event);elapsed=(time.perf_counter_ns()-t)/1e6
        if i>=100:times.append(elapsed)
    result={'hardware':hardware(),'fresh_process_import_rss_mib':baseline,'final_process_rss_mib':rss_mib(),'models':measurements,'raw_event_pipeline':{'batch':1,'p50_ms':float(np.percentile(times,50)),'p90_ms':float(np.percentile(times,90)),'p95_ms':float(np.percentile(times,95)),'samples_per_second':float(1000/np.mean(times))},'method':'CPU, 100 warmups and 1000 calls. Model timings include scaling and network scoring on precomputed features; exclude history, chemistry, explanations, I/O and networking. Raw-event pipeline includes feature derivation, chemical inference, AE score and explanations; prefix restoration and file loading are outside timing. RAM is process-wide, includes runtime/data/all loaded states.'}
    (ROOT/'results/efficiency.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
