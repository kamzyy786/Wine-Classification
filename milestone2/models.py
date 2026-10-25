"""Small PyTorch models and portable runtime measurements."""
import os,platform
from pathlib import Path
import numpy as np
import torch
from torch import nn

def autoencoder(d,bottleneck=3):
    return nn.Sequential(nn.Linear(d,8),nn.ReLU(),nn.Linear(8,bottleneck),nn.ReLU(),nn.Linear(bottleneck,8),nn.ReLU(),nn.Linear(8,d))

def classifier(d):return nn.Sequential(nn.Linear(d,16),nn.ReLU(),nn.Linear(16,2))

def rss_mib():
    if Path('/proc/self/status').exists():return int(next(l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:')))/1024
    import psutil
    return psutil.Process().memory_info().rss/2**20

def hardware():
    cpu=platform.processor()
    if Path('/proc/cpuinfo').exists():cpu=next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name'))
    if Path('/proc/meminfo').exists():ram=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal:')))/2**20
    else:
        import psutil
        ram=psutil.virtual_memory().total/2**30
    return {'cpu':cpu,'logical_cpus':os.cpu_count(),'ram_gib':ram,'torch_threads':torch.get_num_threads(),'python':platform.python_version(),'pytorch':torch.__version__,'device':'CPU; no GPU used'}

def score(model,x,is_ae):
    with torch.inference_mode():
        t=torch.as_tensor(x,dtype=torch.float32);output=model(t)
        return ((t-output)**2).mean(1).numpy() if is_ae else output.softmax(1)[:,1].numpy()

def dense_flops(model):return 2*sum(m.in_features*m.out_features for m in model.modules() if isinstance(m,nn.Linear))
