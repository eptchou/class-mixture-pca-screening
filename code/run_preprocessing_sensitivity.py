"""Run the controlled MF preprocessing-order comparison.

Usage: python run_preprocessing_sensitivity.py before|after [seeds...]
The 'before' arm writes results/sensitivity/mfeat_before; 'after' uses
results/mfeat_r0.1. ADAC_RESULTS_DIR can select a fresh output directory.
Existing seed CSVs are skipped, as in run_condition.py.
"""
import sys, runpy
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import engine as E

order=sys.argv[1];seeds=sys.argv[2:] or [str(i) for i in range(1,11)]
if order not in ('before','after'):raise ValueError('Choose before or after')

def before(name,ratio,seed):
    if name!='mfeat' or ratio!=.1:raise ValueError('This sensitivity is MF at rho=0.1 only')
    d=np.load(Path(E.DATA)/'mfeat.npz');X=d['X'].astype(np.float32);y=d['y']
    tv,te,ytv,yte=train_test_split(X,y,test_size=.1,stratify=y,random_state=42)
    tr,va,ytr,yva=train_test_split(tv,ytv,test_size=.3,stratify=ytv,random_state=42)
    sc=StandardScaler().fit(tr)
    tr,va,te=(sc.transform(a).astype(np.float32) for a in (tr,va,te))
    tb,ytb=te,yte;vb,yvb=va,yva
    rng=np.random.default_rng(1000+seed)
    tr,ytr=E.downsample_class(tr,ytr,0,.1,rng)
    va,yva=E.downsample_class(va,yva,0,.1,rng)
    te,yte=E.downsample_class(te,yte,0,.1,rng)
    return dict(Xtr=tr,ytr=ytr,Xva=va,yva=yva,Xte=te,yte=yte,
                Xte_bal=tb,yte_bal=ytb,Xva_bal=vb,yva_bal=yvb,
                minority=0,kind='numeric',C=10)
if order=='before':
    E.load_condition=before
    E.RESULTS=str(Path(E.RESULTS)/'sensitivity');tag='mfeat_before'
else:tag='mfeat_r0.1'
sys.argv=['run_condition.py','mfeat','0.1',tag,*seeds]
runpy.run_path(str(Path(__file__).resolve().parent/'run_condition.py'),run_name='__main__')
