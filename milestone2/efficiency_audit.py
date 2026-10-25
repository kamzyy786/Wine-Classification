"""Fresh-process inference footprint and explicit dense training-compute estimates."""
import json, time
import numpy as np
import pandas as pd
from demo import Screener
from experiment import ROOT, TX, rss, peak_rss_mib

def main():
    baseline=rss()/2**20
    screen=Screener(); loaded=rss()/2**20
    tx=pd.read_csv(ROOT/'data/synthetic_test.csv'); wine=pd.read_csv(ROOT/'data/wine.csv').drop(columns='target')
    row=tx.iloc[0]; args=(wine.iloc[int(row.sample_id)].to_numpy(),int(row.declared_cultivar),row[TX].to_numpy())
    for _ in range(100):screen.predict(*args)
    durations=[]
    for _ in range(1000):
        t=time.perf_counter_ns();screen.predict(*args);durations.append((time.perf_counter_ns()-t)/1e6)
    r=json.loads((ROOT/'results/metrics.json').read_text()); w=r['chemical_selected_width']; h=16
    chemical_forward=2*(13*w+w*3);risk_forward=2*(6*h+h*2)
    # For dense affine layers, estimate backward matrix work as twice forward work.
    # ReLU, losses, optimizer, validation and CV/OOF are explicitly excluded.
    chem_training=3*chemical_forward*124*r['chemical_training']['epochs']
    risk_training=3*risk_forward*2480*r['risk_training'][r['selected']]['epochs']
    result={'fresh_process_baseline_rss_mib':baseline,'checkpoint_loaded_rss_mib':loaded,'warm_inference_rss_mib':rss()/2**20,'checkpoint_load_increment_mib':loaded-baseline,'os_peak_rss_mib':peak_rss_mib(),'scoring_and_explanations_batch1':{'warmup':100,'calls':1000,'p50_ms':float(np.percentile(durations,50)),'p90_ms':float(np.percentile(durations,90)),'p95_ms':float(np.percentile(durations,95)),'samples_per_second':float(1000/np.mean(durations))},'training_dense_compute_estimate':{'chemical_gflops':chem_training/1e9,'selected_risk_gflops':risk_training/1e9,'total_main_models_gflops':(chem_training+risk_training)/1e9,'scope':'Analytical dense forward/backward estimate: 3 x dense forward FLOPs x training samples x executed epochs. Includes patience epochs; excludes optimizer, loss, activation, validation, cross-validation and out-of-fold fits. Not hardware profiling.'}}
    (ROOT/'results/efficiency_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
