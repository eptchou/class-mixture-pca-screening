"""Twenty CPU ranking timings, including forward activations (seed 1).

The forward computation is warmed once. Thread limits and CPU affinity
can be set by the caller. This benchmark does not time Siamese fitting.
"""
import os,time,json,platform
from pathlib import Path
import numpy as np,pandas as pd,jax.numpy as jnp
import engine as E
R=Path(E.RESULTS)/'fmnist_r0.1';z=np.load(R/'model_seed1.npz')
p={k:jnp.asarray(z[k]) for k in ['W1','b1','g','be','W2','b2','W3','b3']}
s=dict(mean=jnp.asarray(z['bn_mean']),var=jnp.asarray(z['bn_var']))
D=E.load_condition('fmnist',.1,1);X=D['Xva'];y=D['yva'];E.activations(p,s,X)
rows=[]
for rep in range(20):
 row={'rep':rep}
 for c in ['PCA','CB-PCA','NISP','Variance']:
  t=time.perf_counter();A,B=E.activations(p,s,X)
  if c=='PCA':E.pca_score_from_cov(E.pooled_cov(A))
  elif c=='CB-PCA':E.pca_score_from_cov(E.balanced_cov(A,y,np.arange(D['C'])))
  elif c=='Variance':np.var(A,axis=0,ddof=1)
  else:np.abs(np.asarray(p['W2']))@E.inf_fs_scores(B,.2,np.random.default_rng(0))
  row[c]=time.perf_counter()-t
 rows.append(row)
df=pd.DataFrame(rows);df.to_csv(R/'timing.csv',index=False)
meta=dict(platform=platform.platform(),python=platform.python_version(),n_validation=len(y),
          repetitions=20,cpu_affinity=sorted(os.sched_getaffinity(0)),
          thread_limits={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
          includes='forward activations and score calculation; warmed forward computation',
          median_seconds=df.drop(columns='rep').median().to_dict())
(R/'timing_environment.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
