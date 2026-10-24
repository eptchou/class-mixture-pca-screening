"""Regenerate baseline-scale and preprocessing-order sensitivity tables.

Uses saved seed-level results. No models are fitted by this script.
"""
from pathlib import Path
import os, json, glob
import numpy as np
import pandas as pd
from scipy import stats
R=Path(os.environ.get('ADAC_RESULTS_DIR',str(Path(__file__).resolve().parent/'results')))
P=Path(os.environ.get('ADAC_PAPER_DIR',str(Path(__file__).resolve().parents[1]/'source_multifile')))/'tables'
P.mkdir(parents=True,exist_ok=True)
KS=[1,2,3,4,6,8,10,12,16,20,24,32]
NAMES={'fmnist_bal':'F-MNIST, balanced','fmnist_r0.1':r'F-MNIST, $\rho=0.1$',
       'fmnist_r0.002':r'F-MNIST, $\rho=0.002$','mfeat_r0.1':r'MF, $\rho=0.1$'}
def load(folder):
 d=pd.concat([pd.read_csv(f) for f in sorted(folder.glob('retention_seed*.csv'))],ignore_index=True)
 assert d.seed.nunique()==10 and len(d)==760,folder
 full=d[d.criterion=='Full'].set_index('seed')
 d=d[d.criterion!='Full'].copy()
 d['prr_f1']=d.pair_f1/d.seed.map(full.pair_f1)
 d['rel_ba']=d.bal_acc/d.seed.map(full.bal_acc)
 d['excess_f1']=(d.pair_f1-2/3)/(d.seed.map(full.pair_f1)-2/3)
 d['excess_ba']=(d.bal_acc-.1)/(d.seed.map(full.bal_acc)-.1)
 return d,full
out={'baseline':{},'preprocessing':{}}
with (P/'tab_baseline.tex').open('w') as f:
 for t,n in NAMES.items():
  d,full=load(R/t);z=d[(d.criterion=='PCA') & d.K.isin(KS)]
  means=z[['prr_f1','rel_ba','excess_f1','excess_ba']].mean()
  out['baseline'][t]={k:float(v) for k,v in means.items()}
  f.write(n+' & '+' & '.join(f'{v:.3f}' for v in means)+r' \\'+'\n')
def holm(ps):
 ps=np.asarray(ps);o=np.argsort(ps);q=np.empty(len(ps));v=0
 for j,i in enumerate(o):v=max(v,(len(ps)-j)*ps[i]);q[i]=min(1.,v)
 return q
def test(d,m):
 z=d[d.K.isin(KS)].groupby(['seed','criterion'])[m].mean().unstack('criterion')
 others=['CB-PCA','Variance','NISP','Random'];ps=[]
 for c in others:
  x=z[c]-z.PCA;ps.append(stats.wilcoxon(x).pvalue if np.any(x) else 1.)
 x=z['CB-PCA']-z.PCA
 return x, float(holm(ps)[0])
a,_=load(R/'sensitivity/mfeat_before');b,_=load(R/'mfeat_r0.1')
M={'prr_f1':'PRR','min_recall':'Minority recall','bal_acc':'Balanced accuracy'}
ps=[]
for m in M:
 x,pa=test(a,m);y,pb=test(b,m);dd=y-x
 p=stats.wilcoxon(dd).pvalue if np.any(dd) else 1.
 ci=stats.t.ppf(.975,9)*y.std(ddof=1)/np.sqrt(10)
 out['preprocessing'][m]=dict(before=float(x.mean()),after=float(y.mean()),p_before=pa,p_after=pb,
    change=float(dd.mean()),p_change_raw=float(p),after_ci_unadjusted=[float(y.mean()-ci),float(y.mean()+ci)])
 ps.append(p)
with (P/'tab_preprocessing.tex').open('w') as f:
 for m,q in zip(M,holm(ps)):
  v=out['preprocessing'][m];v['p_change_holm']=float(q)
  f.write(M[m]+f" & {v['before']:+.3f} ({v['p_before']:.3f}) & {v['after']:+.3f} ({v['p_after']:.3f}) & {v['change']:+.3f} ({q:.3f})"+r' \\'+'\n')
(R/'key_numbers_revision.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
