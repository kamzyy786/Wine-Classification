"""Reproducible chemical-aware transaction screening. All transactions are synthetic."""
from pathlib import Path
import json, time, copy, platform, hashlib, threading, os
import numpy as np
import pandas as pd
import torch
from torch import nn
import sklearn
try:
    import resource
except ImportError:
    resource=None
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'; MODELS=ROOT/'models'
torch.set_num_threads(1)
SEED=42
CONFIG={'version':'2.0','seed':42,'chemical_widths':[8,16,32], 'chem_epochs':250,'risk_epochs':200,'patience':25,'learning_rate':0.01,'weight_decay':0.001,'batch_size':32,'transactions_per_sample':20,'risk_threshold_grid':list(np.linspace(.1,.9,33))}

def rss():
    if Path('/proc/self/status').exists():
        return int(next(l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:')))*1024
    import psutil
    return psutil.Process().memory_info().rss

def peak_rss_mib():
    if resource is not None:
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value/(2**20 if platform.system()=='Darwin' else 1024)
    import psutil
    m=psutil.Process().memory_info()
    return getattr(m,'peak_wset',m.rss)/2**20

def memory_gib():
    if Path('/proc/meminfo').exists():
        return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal:')))/2**20
    import psutil
    return psutil.virtual_memory().total/2**30

def network(d,h,k):
    return nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,k)) if h else nn.Sequential(nn.Linear(d,k))

def probs(model,x):
    with torch.inference_mode(): return model(torch.tensor(x,dtype=torch.float32)).softmax(1).numpy()

def fit(x,y,vx,vy,h,k,weighted=False,epochs=200,seed=42,patience=25):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    m=network(x.shape[1],h,k); opt=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=.001)
    weights=len(y)/(k*np.maximum(np.bincount(y,minlength=k),1))
    loss=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32) if weighted else None)
    xt=torch.tensor(x,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.long)
    best=1e30; wait=0; history=[]; times=[]; best_epoch=0
    start=time.perf_counter()
    for e in range(epochs):
        t=time.perf_counter(); m.train()
        for ids in np.array_split(rng.permutation(len(y)),max(1,int(np.ceil(len(y)/32)))):
            opt.zero_grad(); l=loss(m(xt[ids]),yt[ids]); l.backward(); opt.step()
        m.eval(); p=probs(m,vx); tr=probs(m,x)
        vl=float(-np.log(np.clip(p[np.arange(len(vy)),vy],1e-9,1)).mean())
        history.append([e+1,float(-np.log(np.clip(tr[np.arange(len(y)),y],1e-9,1)).mean()),vl,f1_score(vy,p.argmax(1),average='macro')])
        times.append(time.perf_counter()-t)
        if vl<best-1e-5: best=vl; state=copy.deepcopy(m.state_dict()); wait=0; best_epoch=e+1
        else: wait+=1
        if wait>=patience: break
    m.load_state_dict(state); m.eval()
    return m,{'seconds':time.perf_counter()-start,'epochs':len(history),'best_epoch':best_epoch,'seconds_per_epoch':float(np.mean(times)),'history':history}

def metrics(y,p,threshold=None):
    pred=p.argmax(1) if threshold is None else (p[:,1]>=threshold).astype(int)
    pr,re,f,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
    result={'accuracy':accuracy_score(y,pred),'precision_macro':pr,'recall_macro':re,'f1_macro':f,'f1_micro':f1_score(y,pred,average='micro'),'confusion_matrix':confusion_matrix(y,pred).tolist()}
    if p.shape[1]==2:
        a,b,c,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
        result.update(precision_positive=a,recall_positive=b,f1_positive=c,auroc=roc_auc_score(y,p[:,1]),pr_auc=average_precision_score(y,p[:,1]))
    else:
        result.update(auroc=roc_auc_score(y,p,multi_class='ovr',average='macro'),pr_auc=average_precision_score(np.eye(p.shape[1])[y],p,average='macro'))
    return result

def preprocess(x,train):
    imp=SimpleImputer(strategy='median').fit(x[train]); sc=StandardScaler().fit(imp.transform(x[train]))
    return sc.transform(imp.transform(x)).astype('float32'),imp,sc

def chemical_features(p,declared):
    declared_p=p[np.arange(len(p)),declared]
    return np.column_stack([1-declared_p,(p.argmax(1)!=declared).astype(float)])

def generate(ids,y,seed):
    r=np.random.default_rng(seed); rows=[]
    for i in ids:
        for j in range(20):
            kind=r.choice(['normal','label_mismatch','financial','combined'],p=[.80,.08,.08,.04])
            mismatch=kind in ['label_mismatch','combined']; finance=kind in ['financial','combined']
            # Small incidental label errors among normal purchases prevent a perfect rule.
            change=r.random()<(.92 if mismatch else .02)
            declared=int((y[i]+r.integers(1,3))%3) if change else int(y[i])
            price=float(r.lognormal(np.log(25),.38)); quantity=int(r.integers(1,13)); rapid=int(r.poisson(1)); new=int(r.random()<.15)
            if finance:
                if r.random()<.75: price*=float(r.choice([.25,3.5]))
                if r.random()<.70: quantity+=int(r.integers(10,40))
                rapid+=int(r.integers(2,9)); new=int(r.random()<.65)
            rows.append([f'SYN-{seed}-{i}-{j}',int(i),declared,price,quantity,rapid,new,int(kind!='normal'),kind])
    return pd.DataFrame(rows,columns=['transaction_id','sample_id','declared_cultivar','unit_price','quantity','purchases_last_hour','new_merchant_category','anomaly','scenario'])

TX=['unit_price','quantity','purchases_last_hour','new_merchant_category']
ALL=TX+['chemical_disagreement_probability','cultivar_mismatch']

def benchmark(fn,raw,declared,tx,batch=1,n=1000):
    idx=np.arange(batch)%len(raw); a,b,c=raw[idx],declared[idx],tx[idx]
    for _ in range(100): fn(a,b,c)
    durations=[]
    for _ in range(n):
        t=time.perf_counter_ns(); fn(a,b,c); durations.append((time.perf_counter_ns()-t)/1e6)
    return {'batch':batch,'p50_ms':float(np.percentile(durations,50)),'p90_ms':float(np.percentile(durations,90)),'p95_ms':float(np.percentile(durations,95)),'samples_per_second':batch/(np.mean(durations)/1000)}

def main():
    OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    start=time.perf_counter(); peak=[rss()]; stop=threading.Event()
    def monitor():
        while not stop.wait(.005): peak[0]=max(peak[0],rss())
    thread=threading.Thread(target=monitor,daemon=True); thread.start()
    df=pd.read_csv(ROOT/'data/wine.csv'); x=df.drop(columns='target').to_numpy(); y=df.target.to_numpy(); ids=np.arange(len(y))
    tr,rest=train_test_split(ids,test_size=.3,stratify=y,random_state=42)
    va,te=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=42)
    for name,idx in [('train',tr),('validation',va),('test',te)]: pd.DataFrame({'sample_id':idx,'target':y[idx]}).to_csv(OUT/f'{name}_sample_ids.csv',index=False)
    folds=list(StratifiedKFold(3,shuffle=True,random_state=42).split(tr,y[tr])); cv=[]
    for width in CONFIG['chemical_widths']:
        scores=[]
        for fi,(a,b) in enumerate(folds):
            z,_,_=preprocess(x,tr[a]); m,log=fit(z[tr[a]],y[tr[a]],z[tr[b]],y[tr[b]],width,3,epochs=250,seed=42+fi)
            scores.append(f1_score(y[tr[b]],probs(m,z[tr[b]]).argmax(1),average='macro'))
        cv.append({'width':width,'fold_f1':scores,'mean_f1':float(np.mean(scores))})
    width=max(cv,key=lambda a:(a['mean_f1'],-a['width']))['width']
    z,imp,sc=preprocess(x,tr)
    chem,chem_log=fit(z[tr],y[tr],z[va],y[va],width,3,epochs=250)
    cp=probs(chem,z)
    # OOF chemical predictions protect risk training from in-sample chemical accuracy.
    for fi,(a,b) in enumerate(folds):
        aa,av=train_test_split(tr[a],test_size=.2,stratify=y[tr[a]],random_state=42+fi)
        zz,_,_=preprocess(x,aa); m,_=fit(zz[aa],y[aa],zz[av],y[av],width,3,epochs=250,seed=42+fi)
        cp[tr[b]]=probs(m,zz[tr[b]])
    sets={}; features={}
    for j,(name,idx) in enumerate([('train',tr),('validation',va),('test',te)]):
        tx=generate(idx,y,100+j); tx.to_csv(ROOT/'data'/f'synthetic_{name}.csv',index=False); sets[name]=tx
        features[name]=np.column_stack([tx[TX].to_numpy(),chemical_features(cp[tx.sample_id],tx.declared_cultivar.to_numpy())]).astype('float32')
    assert not set(sets['train'].sample_id)&set(sets['test'].sample_id)
    candidates=[]; models={}; logs={}; scalers={}
    for name,cols,h,weighted in [('transaction_only',[0,1,2,3],16,True),('combined_linear',list(range(6)),0,True),('combined_mlp',list(range(6)),16,True),('combined_unweighted',list(range(6)),16,False)]:
        scale=StandardScaler().fit(features['train'][:,cols]); a=scale.transform(features['train'][:,cols]); b=scale.transform(features['validation'][:,cols])
        model,log=fit(a,sets['train'].anomaly.to_numpy(),b,sets['validation'].anomaly.to_numpy(),h,2,weighted=weighted)
        p=probs(model,b); yy=sets['validation'].anomaly.to_numpy()
        threshold=max(CONFIG['risk_threshold_grid'],key=lambda t:f1_score(yy,p[:,1]>=t,average='macro'))
        item={'name':name,'columns':cols,'hidden':h,'weighted':weighted,'threshold':threshold,'validation':metrics(yy,p,threshold)}
        candidates.append(item); models[name]=model; logs[name]=log; scalers[name]=scale
    # Selection frozen before any test performance is calculated.
    selected=max([c for c in candidates if c['name'] in ['combined_linear','combined_mlp']],key=lambda c:c['validation']['f1_macro'])
    (OUT/'selection.json').write_text(json.dumps(selected,indent=2))
    for item in candidates:
        name=item['name']; testp=probs(models[name],scalers[name].transform(features['test'][:,item['columns']]))
        item['test']=metrics(sets['test'].anomaly.to_numpy(),testp,item['threshold'])
        pred=(testp[:,1]>=item['threshold']).astype(int)
        item['test_scenario_flag_rates']={k:float(pred[sets['test'].scenario==k].mean()) for k in sets['test'].scenario.unique()}
        pd.DataFrame({'transaction_id':sets['test'].transaction_id,'probability':testp[:,1],'prediction':pred,'actual':sets['test'].anomaly,'scenario':sets['test'].scenario}).to_csv(OUT/f'{name}_test_predictions.csv',index=False)
    chem_metrics={'validation':metrics(y[va],probs(chem,z[va])),'test':metrics(y[te],probs(chem,z[te])),'majority_test_accuracy':float((y[te]==np.bincount(y[tr]).argmax()).mean())}
    name=selected['name']; risk=models[name]; rs=scalers[name]; cols=selected['columns']
    def pipeline(raw,declared,tx):
        pp=probs(chem,sc.transform(imp.transform(raw))); ff=np.column_stack([tx,chemical_features(pp,declared)])[:,cols]
        return probs(risk,rs.transform(ff))
    raw=x[sets['test'].sample_id]; declared=sets['test'].declared_cultivar.to_numpy(); tx=sets['test'][TX].to_numpy()
    latency=[benchmark(pipeline,raw,declared,tx,batch=b) for b in [1,32]]
    inference_rss=rss()/2**20
    bundle={'chem_state':chem.state_dict(),'risk_state':risk.state_dict(),'chemical_width':width,'risk_width':selected['hidden'],'risk_columns':cols,'threshold':float(selected['threshold']),'features':df.drop(columns='target').columns.tolist(),'median':imp.statistics_.tolist(),'chemical_mean':sc.mean_.tolist(),'chemical_scale':sc.scale_.tolist(),'risk_mean':rs.mean_.tolist(),'risk_scale':rs.scale_.tolist()}
    torch.save(bundle,MODELS/'screening.pt')
    params=sum(p.numel() for p in chem.parameters())+sum(p.numel() for p in risk.parameters())
    # Dense arithmetic estimate: 2 FLOPs per multiply-accumulate; biases/ReLU/softmax excluded.
    flops=2*(13*width+width*3+(len(cols)*selected['hidden']+selected['hidden']*2 if selected['hidden'] else len(cols)*2))
    for label,log in [('chemical',chem_log)]+list(logs.items()):
        hist=pd.DataFrame(log.pop('history'),columns=['epoch','train_loss','validation_loss','validation_macro_f1']); hist.to_csv(OUT/f'{label}_learning.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(9,3)); axes[0].plot(hist.epoch,hist.train_loss,label='Training'); axes[0].plot(hist.epoch,hist.validation_loss,label='Validation'); axes[0].set(xlabel='Epoch',ylabel='Cross entropy'); axes[0].legend(); axes[1].plot(hist.epoch,hist.validation_macro_f1); axes[1].set(xlabel='Epoch',ylabel='Validation macro F1 (argmax)'); fig.tight_layout(); fig.savefig(OUT/f'{label}_learning.png',dpi=160); plt.close(fig)
    stop.set(); thread.join()
    cpu=next((s.split(':',1)[1].strip() for s in (Path('/proc/cpuinfo').read_text().splitlines() if Path('/proc/cpuinfo').exists() else ['model name: '+platform.processor()]) if s.startswith('model name')),'unknown')
    result={'config':CONFIG,'chemical_cv':cv,'chemical_selected_width':width,'chemical_metrics':chem_metrics,'chemical_training':chem_log,'risk_candidates':candidates,'risk_training':logs,'selected':name,'latency':latency,'parameters':params,'estimated_dense_flops_per_sample':flops,'checkpoint_bytes':(MODELS/'screening.pt').stat().st_size,'peak_sampled_process_rss_mib':peak[0]/2**20,'inference_process_rss_mib':inference_rss,'peak_process_rss_mib':peak_rss_mib(),'total_experiment_seconds':time.perf_counter()-start,'environment':{'cpu':cpu,'logical_cpus':os.cpu_count(),'torch_threads':torch.get_num_threads(),'ram_gib':memory_gib(),'python':platform.python_version(),'pytorch':torch.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'gpu':'not used; CPU only'},'source_sha256':hashlib.sha256((ROOT/'data/wine.csv').read_bytes()).hexdigest(),'transactions':{k:{'n':len(v),'anomalies':int(v.anomaly.sum()),'unique_samples':int(v.sample_id.nunique()),'scenarios':v.scenario.value_counts().to_dict()} for k,v in sets.items()}}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2)); (ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2))
    print(json.dumps({k:result[k] for k in ['selected','chemical_metrics','latency','parameters','total_experiment_seconds']},indent=2))

if __name__=='__main__': main()
