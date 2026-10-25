"""Train the history-based implementation; AE fitting never receives anomaly labels."""
from pathlib import Path
import json,time,copy,hashlib,threading,importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,f1_score,roc_auc_score,average_precision_score,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from history import ROOT,FEATURES,BEHAVIOUR,SCENARIOS,generate_events,engineer
from models import autoencoder,classifier,score,rss_mib,hardware,dense_flops
from chemistry import network,probs,fit,preprocess

torch.set_num_threads(1)
CONFIG={'version':'3.0-history-autoencoder','seed':42,'customers':{'train':700,'validation':150,'test':150},'events_per_customer':30,'warmup_events':5,'nominal_anomaly_probability':{'train':.02,'validation':.05,'test':.05},'ae_bottlenecks':[2,3],'ae_hidden_width':8,'threshold_validation_percentile':95,'threshold_sweep_percentiles':[90,95,97.5,99],'optimizer':'Adam','learning_rate':.001,'weight_decay':.0001,'batch_size':256,'max_epochs':150,'patience':15,'min_improvement':.00001,'mlp_width':16}

def train_model(model,x,vx,y=None,vy=None,weighted=False):
    """When y is None, only reconstruction inputs are used for loss and stopping."""
    ae=y is None;torch.manual_seed(42);rng=np.random.default_rng(42)
    xt=torch.tensor(x,dtype=torch.float32);vt=torch.tensor(vx,dtype=torch.float32)
    if not ae:
        yt=torch.tensor(y,dtype=torch.long)
        weights=len(y)/(2*np.bincount(y,minlength=2))
        objective=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32) if weighted else None)
    opt=torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    best=float('inf');wait=0;hist=[];durations=[];begin=time.perf_counter();best_epoch=0
    for epoch in range(CONFIG['max_epochs']):
        t=time.perf_counter();model.train()
        for ids in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/CONFIG['batch_size']))):
            opt.zero_grad();out=model(xt[ids]);loss=((out-xt[ids])**2).mean() if ae else objective(out,yt[ids]);loss.backward();opt.step()
        model.eval()
        with torch.inference_mode():
            tr=model(xt);va=model(vt)
            if ae:
                tl=float(((tr-xt)**2).mean());vl=float(((va-vt)**2).mean());metric=float(np.percentile(((va-vt)**2).mean(1).numpy(),95))
            else:
                tl=float(nn.functional.cross_entropy(tr,yt));vl=float(nn.functional.cross_entropy(va,torch.tensor(vy,dtype=torch.long)));metric=f1_score(vy,va.argmax(1).numpy(),average='macro')
        hist.append([epoch+1,tl,vl,metric]);durations.append(time.perf_counter()-t)
        if vl<best-CONFIG['min_improvement']:best=vl;state=copy.deepcopy(model.state_dict());wait=0;best_epoch=epoch+1
        else:wait+=1
        if wait>=CONFIG['patience']:break
    model.load_state_dict(state);model.eval()
    return {'seconds':time.perf_counter()-begin,'epochs':len(hist),'best_epoch':best_epoch,'seconds_per_epoch':float(np.mean(durations)),'best_validation_loss':best,'history':hist,'parameters':sum(p.numel() for p in model.parameters()),'dense_forward_flops':dense_flops(model),'dense_training_gflops_estimate':3*dense_flops(model)*len(x)*len(hist)/1e9}

def evaluate(y,s,threshold):
    pred=(s>=threshold).astype(int);pr,re,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0);mp,mr,mf,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    return {'accuracy':accuracy_score(y,pred),'precision':pr,'recall':re,'f1_anomaly':f,'precision_macro':mp,'recall_macro':mr,'f1_macro':mf,'f1_micro':f1_score(y,pred,average='micro'),'balanced_accuracy':mr,'auroc':roc_auc_score(y,s),'pr_auc':average_precision_score(y,s),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'false_positive_rate':float(fp/max(1,tn+fp)),'flag_rate':float(pred.mean())}

def main():
    for directory in ['data','models','results']:(ROOT/directory).mkdir(exist_ok=True)
    start=time.perf_counter();peak=[rss_mib()];stop=threading.Event()
    def monitor():
        while not stop.wait(.005):peak[0]=max(peak[0],rss_mib())
    th=threading.Thread(target=monitor,daemon=True);th.start()
    wine=pd.read_csv(ROOT/'data/wine.csv');x=wine.drop(columns='target').to_numpy();y=wine.target.to_numpy();ids=np.arange(len(wine))
    tr,rest=train_test_split(ids,test_size=.3,stratify=y,random_state=42);va,te=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=42)
    splits={'train':tr,'validation':va,'test':te}
    bundle=torch.load(ROOT/'models/chemistry.pt',weights_only=True,map_location='cpu');chem=network(13,bundle['chemical_width'],3);chem.load_state_dict(bundle['chem_state']);chem.eval()
    z=((x-np.array(bundle['chemical_mean']))/np.array(bundle['chemical_scale'])).astype('float32');cp=probs(chem,z)
    # Fixed chemistry architecture from the earlier stage. Each OOF sample is unseen
    # by its fold model and by that model's early-stopping subset.
    folds=StratifiedKFold(3,shuffle=True,random_state=42)
    for fi,(a,b) in enumerate(folds.split(tr,y[tr])):
        aa,av=train_test_split(tr[a],test_size=.2,stratify=y[tr[a]],random_state=42+fi)
        zz,_,_=preprocess(x,aa);m,_=fit(zz[aa],y[aa],zz[av],y[av],bundle['chemical_width'],3,epochs=250,seed=42+fi)
        cp[tr[b]]=probs(m,zz[tr[b]])
    np.save(ROOT/'data/chemical_probabilities.npy',cp)
    frames={};data_summary={}
    for j,(split,idx) in enumerate(splits.items()):
        pd.DataFrame({'sample_id':idx,'target':y[idx]}).to_csv(ROOT/f'data/{split}_wine_ids.csv',index=False)
        events=generate_events(idx,y,CONFIG['customers'][split],100+j,CONFIG['nominal_anomaly_probability'][split],split)
        events.to_csv(ROOT/f'data/events_{split}.csv',index=False)
        frame=engineer(events,cp);frame.to_csv(ROOT/f'data/features_{split}.csv',index=False);frames[split]=frame
        data_summary[split]={'events':len(events),'scored_transactions':len(frame),'customers':frame.customer_id.nunique(),'wine_samples':len(idx),'anomalies':int(frame.anomaly.sum()),'anomaly_fraction':float(frame.anomaly.mean()),'scenario_counts':frame.scenario.value_counts().to_dict()}
    print('Generated 30,000 raw events; history features ready.',flush=True)
    data_ready_seconds=time.perf_counter()-start
    scaler=StandardScaler().fit(frames['train'][FEATURES]);scaled={k:scaler.transform(v[FEATURES]).astype('float32') for k,v in frames.items()}
    models={};logs={};candidates=[]
    for bottleneck in CONFIG['ae_bottlenecks']:
        name=f'ae_combined_b{bottleneck}';torch.manual_seed(42);model=autoencoder(9,bottleneck)
        logs[name]=train_model(model,scaled['train'],scaled['validation']);models[name]=model;candidates.append({'name':name,'bottleneck':bottleneck,'validation_reconstruction_loss':logs[name]['best_validation_loss']})
    chosen=min(candidates,key=lambda d:d['validation_reconstruction_loss']);primary=chosen['name']
    torch.manual_seed(42);models['ae_transaction_only']=autoencoder(7,chosen['bottleneck']);logs['ae_transaction_only']=train_model(models['ae_transaction_only'],scaled['train'][:,:7],scaled['validation'][:,:7])
    # AE architecture and threshold are frozen WITHOUT reading anomaly labels.
    thresholds={name:float(np.percentile(score(model,scaled['validation'][:,:7] if name=='ae_transaction_only' else scaled['validation'],True),CONFIG['threshold_validation_percentile'])) for name,model in models.items()}
    frozen={'primary':primary,'ae_selection':'minimum validation reconstruction MSE; no anomaly labels','primary_bottleneck':chosen['bottleneck'],'thresholds':dict(thresholds),'review_capacity_percentile':95,'ae_candidates':candidates}
    (ROOT/'results/frozen_ae_selection.json').write_text(json.dumps(frozen,indent=2))
    print('AE choices frozen without anomaly labels; training supervised comparisons.',flush=True)
    for weighted in [True,False]:
        name='mlp_weighted' if weighted else 'mlp_unweighted';torch.manual_seed(42);model=classifier(9)
        logs[name]=train_model(model,scaled['train'],scaled['validation'],frames['train'].anomaly.to_numpy(),frames['validation'].anomaly.to_numpy(),weighted)
        models[name]=model;s=score(model,scaled['validation'],False)
        thresholds[name]=float(max(np.linspace(.05,.95,37),key=lambda t:f1_score(frames['validation'].anomaly,s>=t,average='macro')))
    reported=[primary,'ae_transaction_only','mlp_weighted','mlp_unweighted'];results={};scores={};sweeps=[]
    for name in reported:
        is_ae=name.startswith('ae_');cols=7 if name=='ae_transaction_only' else 9;scores[name]={k:score(models[name],v[:,:cols],is_ae) for k,v in scaled.items()}
        results[name]={k:evaluate(frames[k].anomaly.to_numpy(),scores[name][k],thresholds[name]) for k in ['validation','test']}
        pred=scores[name]['test']>=thresholds[name]
        results[name]['test_scenario_recall']={scenario:float(pred[frames['test'].scenario==scenario].mean()) for scenario in SCENARIOS}
        pd.DataFrame({'transaction_id':frames['test'].transaction_id,'customer_id':frames['test'].customer_id,'sample_id':frames['test'].sample_id,'scenario':frames['test'].scenario,'actual':frames['test'].anomaly,'score':scores[name]['test'],'prediction':pred.astype(int)}).to_csv(ROOT/f'results/{name}_test_predictions.csv',index=False)
        if is_ae:
            for percentile in CONFIG['threshold_sweep_percentiles']:
                threshold=float(np.percentile(scores[name]['validation'],percentile));sweeps.append({'model':name,'validation_percentile':percentile,'threshold':threshold,**evaluate(frames['test'].anomaly.to_numpy(),scores[name]['test'],threshold)})
    pd.DataFrame(sweeps).to_csv(ROOT/'results/threshold_tradeoffs.csv',index=False)
    exports={'version':'3.0','features':FEATURES,'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'primary':primary,'bottleneck':chosen['bottleneck'],'thresholds':thresholds,'states':{name:models[name].state_dict() for name in reported}}
    torch.save(exports,ROOT/'models/detectors.pt')
    with torch.inference_mode():
        v=torch.tensor(scaled['test']);error=((v-models[primary](v))**2).numpy()
    explanations=[]
    for i in np.where(scores[primary]['test']>=thresholds[primary])[0]:
        top=np.argsort(error[i])[::-1][:3];row=frames['test'].iloc[i]
        explanations.append({'transaction_id':row.transaction_id,'scenario':row.scenario,'actual':int(row.anomaly),'score':float(scores[primary]['test'][i]),'threshold':thresholds[primary],'top_factors':[{'feature':FEATURES[j],'squared_standardized_error':float(error[i,j]),'share_of_total_error':float(error[i,j]/max(error[i].sum(),1e-12))} for j in top]})
    (ROOT/'results/flag_explanations.json').write_text(json.dumps(explanations,indent=2))
    for name,log in logs.items():
        history=pd.DataFrame(log.pop('history'),columns=['epoch','train_loss','validation_loss','validation_metric']);history.to_csv(ROOT/f'results/{name}_learning.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(9,3));axes[0].plot(history.epoch,history.train_loss,label='Train');axes[0].plot(history.epoch,history.validation_loss,label='Validation');axes[0].set(xlabel='Epoch',ylabel='Reconstruction MSE' if name.startswith('ae') else 'Cross entropy');axes[0].legend();axes[1].plot(history.epoch,history.validation_metric);axes[1].set(xlabel='Epoch',ylabel='Validation score p95' if name.startswith('ae') else 'Validation macro F1');fig.tight_layout();fig.savefig(ROOT/f'results/{name}_learning.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3))
    for name in [primary,'ae_transaction_only']:
        subset=[v for v in sweeps if v['model']==name];ax.plot([v['fp'] for v in subset],[v['fn'] for v in subset],'o-',label=name)
    ax.set(xlabel='False positives',ylabel='Missed anomalies');ax.legend();fig.tight_layout();fig.savefig(ROOT/'results/threshold_tradeoffs.png',dpi=150);plt.close(fig)
    stop.set();th.join()
    result={'config':CONFIG,'primary':primary,'selection':frozen,'data':data_summary,'results':results,'training':logs,'thresholds':thresholds,'environment':hardware(),'peak_training_process_rss_mib':peak[0],'data_and_chemical_oof_seconds':data_ready_seconds,'total_experiment_seconds':time.perf_counter()-start,'checkpoint_bytes':(ROOT/'models/detectors.pt').stat().st_size,'chemistry_checkpoint_sha256':hashlib.sha256((ROOT/'models/chemistry.pt').read_bytes()).hexdigest(),'chemistry_reference':json.loads((ROOT/'results/previous_run_reference.json').read_text())['chemical_metrics'],'source_csv_sha256':hashlib.sha256((ROOT/'data/wine.csv').read_bytes()).hexdigest(),'explanations_count':len(explanations)}
    (ROOT/'results/metrics.json').write_text(json.dumps(result,indent=2));(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2))
    packages=['torch','numpy','pandas','scikit-learn','matplotlib','reportlab'];(ROOT/'environment_versions.txt').write_text('\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in packages)+'\n')
    print(json.dumps({'primary':primary,'data':data_summary,'test':{name:results[name]['test'] for name in reported},'seconds':result['total_experiment_seconds']},indent=2))

if __name__=='__main__':main()
