"""Reconstruct, rather than claim contemporaneous recording of, split indices.

No classifier or Siamese network is trained. Index positions are 0-based.
"""
from pathlib import Path
import ast,json,pickle,hashlib
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split

P=Path(__file__).resolve().parent/'ADAC_Numerical_Audit_Materials_20261024'
if not P.exists():P=Path(__file__).resolve().parents[1]
CODE=P/'code';D=CODE/'data';R=CODE/'results';OUT=P/'indices_reconstructed';OUT.mkdir(exist_ok=True)
TAGS={'fmnist_bal':('fmnist',1.),'fmnist_r0.1':('fmnist',.1),'mfeat_bal':('mfeat',1.),'mfeat_r0.1':('mfeat',.1),'reuters_bal':('reuters_bal',1.),'reuters_nat':('reuters_nat',1.),**{f'fmnist_r{r}':('fmnist',r) for r in [.3,.02,.01,.002]}}
def getfunc(file,name):
 node=next(n for n in ast.parse(file.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
 module=ast.Module(body=[node],type_ignores=[]);scope={'np':np}
 exec(compile(module,str(file),'exec'),scope);return scope[name]
downsample=getfunc(CODE/'engine.py','downsample_class')
halves=getfunc(CODE/'diagnostics.py','halves')
ys={k:np.load(D/(k+'.npz'))['y'] for k in ['fmnist','mfeat']}
reuters=pickle.load((D/'reuters.pkl').open('rb'));lab=np.array(reuters['first_cat'])
counts={c:int((lab==c).sum()) for c in np.unique(lab)}
top3=sorted(counts,key=lambda c:-counts[c])[:3];names=['crude','acq','earn','Others']
ys['reuters']=np.array([names.index(c if c in top3 else 'Others') for c in lab],dtype=np.int64)
summary=[];files=[]
for tag,(name,ratio) in TAGS.items():
 source='reuters' if name.startswith('reuters') else name
 y0=ys[source];ids0=np.arange(len(y0),dtype=np.uint32);ids=ids0.copy();y=y0.copy()
 if name=='reuters_bal':
  rng=np.random.default_rng(42);m=np.bincount(y).min()
  keep=np.concatenate([rng.choice(np.where(y==c)[0],m,replace=False) for c in range(4)])
  rng.shuffle(keep);ids,y=ids[keep],y[keep]
 tv,te,ytv,yte=train_test_split(ids,y,test_size=.1,stratify=y,random_state=42)
 tr,va,ytr,yva=train_test_split(tv,ytv,test_size=.3,stratify=ytv,random_state=42)
 for seed in range(1,11):
  tt,vv,ee=tr.copy(),va.copy(),te.copy();ty,vy,ey=ytr.copy(),yva.copy(),yte.copy()
  if ratio<1:
   rng=np.random.default_rng(1000+seed)
   tt,ty=downsample(tt,ty,0,ratio,rng);vv,vy=downsample(vv,vy,0,ratio,rng);ee,ey=downsample(ee,ey,0,ratio,rng)
  meta=json.loads((R/tag/f'meta_seed{seed}.json').read_text());C=len(np.unique(y))
  assert np.array_equal(np.bincount(vy,minlength=C),meta['class_counts_val']),(tag,seed)
  original=pd.read_csv(R/tag/f'stab_split_seed{seed}.csv').set_index('split')
  z={'validation_source_row_ids':vv,'validation_y':vy.astype(np.uint8),'training_source_row_ids':tt,'test_condition_source_row_ids':ee,'test_before_subsampling_source_row_ids':te,'validation_before_subsampling_source_row_ids':va}
  for rep,a,b in halves(vy,seed,True):
   assert not np.intersect1d(a,b).size
   assert np.array_equal(np.sort(np.r_[a,b]),np.arange(len(vy)))
   ac=np.bincount(vy[a],minlength=C);bc=np.bincount(vy[b],minlength=C)
   assert ac.min()>=2 and bc.min()>=2
   assert ac[0]==original.loc[rep,'minority_A'] and bc[0]==original.loc[rep,'minority_B']
   z[f'split{rep}_A']=a.astype(np.uint16);z[f'split{rep}_B']=b.astype(np.uint16)
   summary.append(dict(tag=tag,seed=seed,split=rep,n_A=len(a),n_B=len(b),minority_A=int(ac[0]),minority_B=int(bc[0]),minimum_per_class_A=int(ac.min()),minimum_per_class_B=int(bc.min()),all_validation_rows_used=True,source='reconstructed_during_packaging_not_original_log'))
  out=OUT/tag/f'seed{seed}.npz';out.parent.mkdir(exist_ok=True);np.savez_compressed(out,**z)
  files.append(dict(tag=tag,seed=seed,path=out.relative_to(P).as_posix(),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),validation_n=len(vy),prepared_dataset=source,origin='deterministic_reconstruction'))
pd.DataFrame(summary).to_csv(OUT/'split_summary.csv',index=False)
pd.DataFrame(files).to_csv(OUT/'index_file_manifest.csv',index=False)
(OUT/'reconstruction_notes.json').write_text(json.dumps({'original_indices_were_saved':False,'method':'Execute the unchanged halves() and downsample_class() definitions extracted by AST from the numerical-run source. Reconstruct data row splits using the saved prepared data and random_state=42. Validate every saved per-class count and every saved minority_A/minority_B count.','zero_based':True,'A_rule':'first floor(n_c/2) after class-specific permutation','B_rule':'remaining ceil(n_c/2); n_c=5 always means A=2, B=3','np_rng_halves':'default_rng(500+seed), shared sequential RNG across ten repeats and ascending class labels','source_row_id_meaning':{'fmnist':'prepared fmnist.npz rows: original official 60000 train rows followed by 10000 official test rows','mfeat':'prepared mfeat.npz rows in the six-block PMLB concatenation order','reuters':'prepared reuters.pkl data order; ids field contains original document ID'},'epistemic_limit':'The original run did not write index arrays or their hashes. These are deterministic reconstructions consistent with the saved code, prepared data, and recorded counts; no contemporaneous index-hash evidence exists.','files':len(files),'stratified_pairs':len(summary)},indent=2))
print(json.dumps({'index_files':len(files),'split_pairs':len(summary),'bytes':sum(p.stat().st_size for p in OUT.rglob('*.npz')),'all_recorded_counts_match':True}))
