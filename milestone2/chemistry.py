"""Chemical-classifier helpers retained from the prior verified run."""
import copy, time
import numpy as np
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
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

def preprocess(x,train):
    imp=SimpleImputer(strategy='median').fit(x[train]); sc=StandardScaler().fit(imp.transform(x[train]))
    return sc.transform(imp.transform(x)).astype('float32'),imp,sc
