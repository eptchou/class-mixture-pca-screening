"""Independent statistical audit of a complete fresh rerun (no model training).

Regenerates seed-level paired estimates, checks result cardinalities, and
compares revised summaries with the previous-version metadata when supplied.
"""
from pathlib import Path
import os,json,hashlib,re
import numpy as np,pandas as pd
from scipy import stats
CODE=Path(__file__).resolve().parent
R=Path(os.environ.get('ADAC_RESULTS_DIR',str(CODE/'results')))
TAGS=['fmnist_bal','fmnist_r0.1','mfeat_bal','mfeat_r0.1','reuters_bal','reuters_nat',
      'fmnist_r0.3','fmnist_r0.02','fmnist_r0.01','fmnist_r0.002']
CRITS=['PCA','CB-PCA','Variance','NISP','Random'];OTHER=CRITS[1:]
KS=[1,2,3,4,6,8,10,12,16,20,24,32];GRID=KS+[48,64,96]
def holm(p):
 p=np.asarray(p);order=np.argsort(p);q=np.empty(len(p));run=0
 for j,i in enumerate(order):run=max(run,(len(p)-j)*p[i]);q[i]=min(1.,run)
 return q
def compare(d,cols):
 out={}
 for m in cols:
  x=d[d.K.isin(KS)].groupby(['seed','criterion'])[m].mean().unstack('criterion').sort_index()
  assert x.shape==(10,5) and x.notna().all().all()
  item={'means':{c:float(x[c].mean()) for c in CRITS},'sd':{c:float(x[c].std(ddof=1)) for c in CRITS},'tests':{}}
  ps=[]
  for c in OTHER:
   v=x[c]-x.PCA;p=stats.wilcoxon(v).pvalue if np.any(v!=0) else 1.;ps.append(p)
   half=stats.t.ppf(.975,9)*v.std(ddof=1)/np.sqrt(10)
   item['tests'][c]={'difference':float(v.mean()),'ci_low':float(v.mean()-half),'ci_high':float(v.mean()+half),'p_raw':float(p)}
  for c,p in zip(OTHER,holm(ps)):item['tests'][c]['p_holm']=float(p)
  out[m]=item
 return out
result={'primary':{},'probe':{},'counts':{},'classifiers':{}};primary_total=0
for tag in TAGS:
 fs=sorted((R/tag).glob('retention_seed*.csv'));assert len(fs)==10,(tag,len(fs))
 frames=[]
 for f in fs:
  d=pd.read_csv(f);assert len(d)==76 and not d.duplicated(['criterion','K']).any(),f
  assert (d.criterion=='Full').sum()==1
  for c in CRITS:assert sorted(d[d.criterion==c].K)==GRID,(f,c)
  assert np.isfinite(d[['pair_f1','min_recall','bal_acc']]).all().all()
  assert d[['pair_f1','min_recall','bal_acc']].ge(0).all().all() and d[['pair_f1','min_recall','bal_acc']].le(1).all().all()
  frames.append(d)
 d=pd.concat(frames,ignore_index=True);full=d[d.criterion=='Full'].set_index('seed');d=d[d.criterion!='Full'].copy()
 d['prr_f1']=d.pair_f1/d.seed.map(full.pair_f1)
 result['primary'][tag]=compare(d,['prr_f1','min_recall','bal_acc'])
 result['primary'][tag]['full']={m:float(full[m].mean()) for m in ['pair_f1','pair_acc','min_recall','bal_acc','macro_f1']}
 primary_total+=sum(len(f) for f in frames)
 # Every same-network stability record uses fixed classes with >=2 per half.
 s=pd.read_csv(R/tag/'stab_split.csv');assert len(s)==100 and s.seed.nunique()==10,tag
 assert s.minority_A.min()>=2 and s.minority_B.min()>=2,tag
 assert len(pd.read_csv(R/tag/'stab_nogueira.csv'))==160
 assert len(list((R/tag).glob('model_seed*.npz')))==10
 result['classifiers'][tag]={'n':10,'minimum_minority_half':int(min(s.minority_A.min(),s.minority_B.min()))}
result['counts']['primary_fits']=primary_total;assert primary_total==7600
for tag in ['fmnist_r0.1','mfeat_r0.1','reuters_nat']:
 fs=sorted((R/tag).glob('probe_seed*.csv'));assert len(fs)==10,(tag,len(fs))
 d=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True);assert len(d)==760
 full=d[d.criterion=='Full'];d=d[d.criterion!='Full']
 result['probe'][tag]=compare(d,['probe_min_recall','probe_bal_acc'])
 result['probe'][tag]['full']={m:float(full[m].mean()) for m in ['probe_min_recall','probe_bal_acc']}
result['counts']['probe_fits']=2280
before=list((R/'sensitivity/mfeat_before').glob('retention_seed*.csv'));assert len(before)==10
assert sum(len(pd.read_csv(f)) for f in before)==760
result['counts']['preprocessing_before_fits']=760
result['counts']['total_siamese_fits']=10640
assert len(pd.read_csv(R/'spectral_check.csv'))==60
assert len(pd.read_csv(R/'nisp_weight_diagnostic.csv'))==100
# Compare with the independently generated aggregate/extra summaries.
k=json.loads((R/'key_numbers.json').read_text());e=json.loads((R/'key_numbers_extra.json').read_text())
for tag in TAGS:
 for m in ['prr_f1','min_recall','bal_acc']:
  for c in OTHER:
   v=result['primary'][tag][m]['tests'][c];u=k[f'test_{tag}_{m}'][c]
   assert abs(v['difference']+u['diff'])<.000051,(tag,m,c,'difference')
   assert abs(v['p_holm']-u['p_holm'])<.000051,(tag,m,c,'p')
prev_path=CODE/'provenance/previous_key_numbers.json'
if prev_path.exists():
 old=json.loads(prev_path.read_text());changes=[]
 for tag in TAGS:
  for m in ['prr_f1','min_recall','bal_acc']:
   for c in OTHER:
    oldp=old[f'test_{tag}_{m}'][c]['p_holm'];new=result['primary'][tag][m]['tests'][c]
    changes.append(dict(condition=tag,measure=m,criterion=c,previous_difference=-old[f'test_{tag}_{m}'][c]['diff'],fresh_difference=new['difference'],previous_p_holm=oldp,fresh_p_holm=new['p_holm'],significance_changed=(oldp<.05)!=(new['p_holm']<.05)))
 pd.DataFrame(changes).to_csv(R/'primary_vs_previous_summary.csv',index=False)
 result['changed_significance']=[c for c in changes if c['significance_changed']]
# Describe cross-condition patterns with the explicitly stated within-family adjustment.
imbal=['fmnist_r0.1','mfeat_r0.1','reuters_nat','fmnist_r0.3','fmnist_r0.02','fmnist_r0.01','fmnist_r0.002']
result['patterns']={
 'cb_minority_positive':[t for t in imbal if result['primary'][t]['min_recall']['tests']['CB-PCA']['difference']>0],
 'cb_minority_significant':[t for t in imbal if result['primary'][t]['min_recall']['tests']['CB-PCA']['p_holm']<.05],
 'cb_balanced_accuracy_significant':[t for t in imbal if result['primary'][t]['bal_acc']['tests']['CB-PCA']['p_holm']<.05],
 'probe_cb_minority_significant':[t for t in result['probe'] if result['probe'][t]['probe_min_recall']['tests']['CB-PCA']['p_holm']<.05],
 'probe_cb_balanced_accuracy_significant':[t for t in result['probe'] if result['probe'][t]['probe_bal_acc']['tests']['CB-PCA']['p_holm']<.05]}
(R/'audit_fresh.json').write_text(json.dumps(result,indent=2));print(json.dumps({'counts':result['counts'],'patterns':result['patterns'],'changed_significance':result.get('changed_significance',[])},indent=2))
