"""Exact sign tests and signed-rank tests from saved seed-level CSVs only."""
import os,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]
R=Path(os.environ.get('ADAC_RESULTS_DIR',str(ROOT/'code/results')))
P=Path(os.environ.get('ADAC_PAPER_DIR',str(ROOT/'rebuilt/source_multifile')))
(P/'tables').mkdir(parents=True,exist_ok=True)
T=['fmnist_r0.3','fmnist_r0.1','fmnist_r0.02','fmnist_r0.01','fmnist_r0.002','mfeat_r0.1','reuters_nat']
O=['CB-PCA','Variance','NISP','Random'];KS=[1,2,3,4,6,8,10,12,16,20,24,32]
def holm(p):
 q=np.empty(len(p));v=0.
 for j,i in enumerate(np.argsort(p)):v=max(v,(len(p)-j)*p[i]);q[i]=min(1.,v)
 return q
rows=[];lines=[]
for tag in T:
 d=pd.concat([pd.read_csv(R/tag/f'retention_seed{s}.csv') for s in range(1,11)])
 z=d[(d.criterion!='Full')&d.K.isin(KS)].groupby(['seed','criterion']).min_recall.mean().unstack()
 rp=[];sp=[];tmp=[]
 for c in O:
  x=z[c]-z.PCA;n=int((x!=0).sum());pos=int((x>0).sum());zero=int((x==0).sum());neg=int((x<0).sum())
  rp.append(float(stats.wilcoxon(x).pvalue) if n else 1.)
  sp.append(float(stats.binomtest(pos,n,.5).pvalue) if n else 1.)
  tmp.append(dict(tag=tag,criterion=c,mean_difference=float(x.mean()),positive=pos,zero=zero,negative=neg))
 for v,p,q,a,b in zip(tmp,rp,sp,holm(rp),holm(sp)):rows.append(dict(**v,wilcoxon_raw=p,sign_raw=q,wilcoxon_holm=float(a),sign_holm=float(b)))
 v=rows[-4]
 name='Reuters, natural' if tag=='reuters_nat' else ('MF' if tag.startswith('mfeat') else 'F-MNIST')+r', $\rho='+tag.split('_r')[1]+'$'
 lines.append(name+f" & {v['positive']}/{v['zero']}/{v['negative']} & {v['mean_difference']:+.3f} & {v['wilcoxon_holm']:.3f} & {v['sign_holm']:.3f}"+r' \\'+'\n')
(P/'tables/tab_sign.tex').write_text(''.join(lines))
pd.DataFrame(rows).to_csv(P/'sign_tests_all_comparisons.csv',index=False)
print(json.dumps({'Wilcoxon_CB_significant':sum(x['criterion']=='CB-PCA' and x['wilcoxon_holm']<.05 for x in rows),'sign_CB_significant':sum(x['criterion']=='CB-PCA' and x['sign_holm']<.05 for x in rows)}))
