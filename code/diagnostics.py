"""Stratified stability and fixed-network diagnostics for the revised manuscript."""
import os,sys,argparse,time,json
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--task',choices=['diagnostics'],default='diagnostics')
ap.add_argument('--cores',default='',help='Optional comma-separated available CPU IDs')
ap.add_argument('--tags',nargs='+',default=None)
ap.add_argument('--seeds',nargs='+',type=int,default=list(range(1,11)))
ap.add_argument('--force',action='store_true',help='Recompute diagnostics; reuse fitted classifiers')
args=ap.parse_args()
if args.cores and hasattr(os,'sched_setaffinity'):
    os.sched_setaffinity(0,{int(c) for c in args.cores.split(',')})
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd,jax.numpy as jnp
from scipy.stats import spearmanr,hypergeom
from scipy.optimize import brentq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,balanced_accuracy_score,f1_score
from functools import lru_cache

ROOT=Path(__file__).resolve().parent
import engine as E
R=Path(E.RESULTS);R.mkdir(exist_ok=True,parents=True)
TAGS={'fmnist_bal':('fmnist',1.),'fmnist_r0.1':('fmnist',.1),'mfeat_bal':('mfeat',1.),'mfeat_r0.1':('mfeat',.1),
      'reuters_bal':('reuters_bal',1.),'reuters_nat':('reuters_nat',1.),
      **{f'fmnist_r{r}':('fmnist',r) for r in [.3,.02,.01,.002]}}
CROSS=list(TAGS)[:6]

def data(tag,seed):
    return E.load_condition(*TAGS[tag],seed)

def model(tag,seed,D):
    f=R/tag/f'model_seed{seed}.npz';f.parent.mkdir(exist_ok=True,parents=True)
    if not f.exists():
        p,s,acc=E.train_classifier(D,seed)
        np.savez_compressed(f,**{k:np.asarray(v) for k,v in p.items()},bn_mean=np.asarray(s['mean']),bn_var=np.asarray(s['var']),val_acc=acc)
    z=np.load(f);p={k:jnp.asarray(z[k]) for k in ['W1','b1','g','be','W2','b2','W3','b3']};s={'mean':jnp.asarray(z['bn_mean']),'var':jnp.asarray(z['bn_var'])}
    return p,s,float(z['val_acc'])


def scores(A,B,y,W2,C,seed):
    S=E.pooled_cov(A);pa,lam=E.pca_score_from_cov(S);cb,_=E.pca_score_from_cov(E.balanced_cov(A,y,np.arange(C)))
    return {'PCA':pa,'Variance':np.diag(S).copy(),'CB-PCA':cb,'NISP':np.abs(np.asarray(W2))@E.inf_fs_scores(B,.2,np.random.default_rng(seed))},lam

def halves(y,seed,stratified=True):
    rng=np.random.default_rng(500+seed)
    for r in range(10):
        if stratified:
            aa=[];bb=[]
            for c in np.unique(y):
                ids=rng.permutation(np.where(y==c)[0]);h=len(ids)//2;aa.append(ids[:h]);bb.append(ids[h:])
            yield r,np.concatenate(aa),np.concatenate(bb)
        else:
            pr=rng.permutation(len(y));h=len(y)//2;yield r,pr[:h],pr[h:2*h]

def jac(a,b):
    a,b=set(a),set(b);return len(a&b)/len(a|b)

@lru_cache(None)
def chance(ka,kb):
    j=np.arange(max(0,ka+kb-128),min(ka,kb)+1)
    return float(np.sum(hypergeom.pmf(j,128,ka,kb)*j/(ka+kb-j)))

def nog(z):
    z=np.array(z);m=len(z);q=z.mean(0);k=z.sum(1).mean()/128
    return 1-np.mean(m/(m-1)*q*(1-q))/(k*(1-k))

def diagnostics():
    for tag in (args.tags or list(TAGS)):
        folder=R/tag;folder.mkdir(exist_ok=True)
        for seed in args.seeds:
            finished=folder/f'diagnostics_seed{seed}.json'
            if finished.exists() and not args.force:continue
            t=time.time();D=data(tag,seed);p,s,acc=model(tag,seed,D)
            A,B=E.activations(p,s,D['Xva']);y=D['yva'];C=D['C']
            sc,lam=scores(A,B,y,p['W2'],C,seed)
            corr={'seed':seed,**{f'rho_PCA_{c}':float(spearmanr(sc['PCA'],sc[c])[0]) for c in ['Variance','CB-PCA','NISP']}}
            norm=np.abs(np.asarray(p['W2'])).sum(1)
            weight_rho=float(spearmanr(norm,sc['NISP'])[0])
            inf_scores=E.inf_fs_scores(B,.2,np.random.default_rng(seed))
            inf_cv=float(inf_scores.std(ddof=1)/inf_scores.mean())
            rows=[];selection={c:{k:[] for k in [8,16,32,64]} for c in sc};opnoise=[];snoise=[]
            for rep,a,b in halves(y,seed):
                row={'seed':seed,'split':rep,'minority_A':int((y[a]==0).sum()),'minority_B':int((y[b]==0).sum())}
                sa,la=scores(A[a],B[a],y[a],p['W2'],C,rep);sb,lb=scores(A[b],B[b],y[b],p['W2'],C,rep)
                for c in sc:
                    oa,ob=E.order(sa[c]),E.order(sb[c]);row[c+'_rho']=float(spearmanr(sa[c],sb[c])[0])
                    for k in [8,16,32,64]:
                        row[c+f'_J{k}']=jac(oa[:k],ob[:k])
                        for o in [oa,ob]:
                            z=np.zeros(128);z[o[:k]]=1;selection[c][k].append(z)
                ta,tb=E.thresholds_from_lam(la),E.thresholds_from_lam(lb)
                oa,ob=E.order(sa['PCA']),E.order(sb['PCA'])
                for tn in ta:
                    ka,kb=ta[tn],tb[tn];row['PCA_thrJ_'+tn]=jac(oa[:ka],ob[:kb]);row['PCA_thrChance_'+tn]=chance(ka,kb);row['PCA_thrK_'+tn]=(ka+kb)/2
                rows.append(row)
                opnoise.append(np.linalg.norm(E.pooled_cov(A[a])-E.pooled_cov(A[b]),2))
                snoise.append(np.abs(sa['PCA']-sb['PCA']).max())
            pd.DataFrame(rows).to_csv(folder/f'stab_split_seed{seed}.csv',index=False)
            nrows=[dict(seed=seed,criterion=c,K=k,phi=nog(selection[c][k])) for c in sc for k in [8,16,32,64]]
            pd.DataFrame(nrows).to_csv(folder/f'stab_nogueira_seed{seed}.csv',index=False)
            # Composition sensitivity: PCA needs no class covariance in random halves.
            rr=[]
            for rep,a,b in halves(y,seed,False):
                xa,_=E.pca_score_from_cov(E.pooled_cov(A[a]));xb,_=E.pca_score_from_cov(E.pooled_cov(A[b]));rr.append(spearmanr(xa,xb)[0])
            strata=dict(seed=seed,PCA_random=float(np.mean(rr)),PCA_stratified=float(np.mean([r['PCA_rho'] for r in rows])))
            # Population assumptions are not established by these sample diagnostics.
            cov=E.pooled_cov(A);w,V=E.eig_desc(cov);trace=float(w.sum());r=int((w>0).sum());positive=w[:r]
            gs=np.minimum(np.r_[np.inf,positive[:-1]]-positive,positive-np.r_[positive[1:],0])
            aa=float(2**1.5*np.sum((positive/trace)/gs));ss=np.sort(sc['PCA'])[::-1]
            spectral=dict(seed=seed,r=r,gmin_rel=float(gs.min()/w[0]),noise_op_median=float(np.median(opnoise)))
            for k in [16,32]:
                gap=float(ss[k-1]-ss[k]);u=brentq(lambda u:2*(u*trace/128)*(aa+256/(trace*(1-u)))-gap,0,1-1e-12,xtol=1e-15)
                eps=u*trace/128;spectral[f'eps{k}']=eps;spectral[f'gap{k}']=gap;spectral[f'ratio{k}']=float(np.median(opnoise)/eps)
            overlaps=[];thr=E.thresholds_from_lam(lam)
            for name,k in thr.items():
                ov=len(set(E.order(sc['PCA'])[:k])&set(E.order(sc['NISP'])[:k]));pe=(k/128)**2+(1-k/128)**2;po=1-2*(k-ov)/128
                overlaps.append(dict(seed=seed,threshold=name,K=k,overlap=ov,expected=k*k/128,p_hyper=float(hypergeom.sf(ov-1,128,k,k)),kappa=(po-pe)/(1-pe)))
            obj={'correlations':corr,'weight_rho':weight_rho,'inf_fs_cv':inf_cv,'stratified':strata,'spectral':spectral,'overlaps':overlaps,'val_acc':acc,'thresholds':E.thresholds_from_lam(lam)}
            if tag=='fmnist_bal':
                mus=np.array([A[y==c].mean(0) for c in range(C)]);scs=[np.cov(A[y==c],rowvar=False,ddof=0) for c in range(C)]
                ac=[v+np.outer(m,m) for v,m in zip(scs,mus)];boundB=max(np.linalg.norm(v,2)+m@m for v,m in zip(scs,mus))
                def sigma(pi):
                    m=pi@mus;return sum(a*v for a,v in zip(pi,ac))-np.outer(m,m)
                p0=np.full(C,1/C);s0=sigma(p0);ss0,lam0=E.pca_score_from_cov(s0);o0=E.order(ss0);k85=E.thresholds_from_lam(lam0)['Var85'];mix=[]
                for rt in [.3,.1,.02,.01,.002]:
                    pi=np.ones(C);pi[0]=rt;pi/=pi.sum();sx=sigma(pi);sv,_=E.pca_score_from_cov(sx);order=E.order(sv)
                    row=dict(seed=seed,ratio=rt,dpi_l1=float(abs(pi-p0).sum()),dSigma_op=float(np.linalg.norm(sx-s0,2)),bound_3B=float(3*boundB*abs(pi-p0).sum()),
                             dscore_max=float(abs(sv-ss0).max()),rho=float(spearmanr(sv,ss0)[0]),noise_split_median=float(np.median(snoise)),K85=k85)
                    sorted0=np.sort(ss0)[::-1]
                    for k,key in [(16,'16'),(32,'32'),(k85,'K85')]:
                        gap=float(sorted0[k-1]-sorted0[k]);row['J_'+key]=jac(order[:k],o0[:k]);row['gap_'+key]=gap;row['guaranteed_'+key]=bool(2*row['dscore_max']<gap)
                    mix.append(row)
                obj['mixture']=mix
            finished.write_text(json.dumps(obj,indent=2))
            meta=folder/f'meta_seed{seed}.json';orig=json.loads(meta.read_text()) if meta.exists() else dict(seed=seed,n_train=len(D['ytr']),n_val=len(D['yva']),n_test=len(D['yte']),class_counts_train=np.bincount(D['ytr']).tolist(),class_counts_test=np.bincount(D['yte']).tolist());orig['val_acc']=acc;orig['thresholds']=E.thresholds_from_lam(lam);orig['class_counts_val']=np.bincount(D['yva'],minlength=C).tolist();meta.write_text(json.dumps(orig,indent=1))
            print(tag,seed,'diagnostics done',round(time.time()-t,1),flush=True)
    # Merge every available seed record. Per-tag or per-seed runs do not
    # discard already computed diagnostics for the other conditions.
    spec=[];weight=[]
    for tag in TAGS:
        f=R/tag
        files=sorted(f.glob('diagnostics_seed*.json'),key=lambda p:int(p.stem.split('seed')[1]))
        if not files:continue
        js=[json.loads(p.read_text()) for p in files]
        seeds=[int(j['correlations']['seed']) for j in js]
        pd.concat([pd.read_csv(f/f'stab_split_seed{s}.csv') for s in seeds]).to_csv(f/'stab_split.csv',index=False)
        pd.concat([pd.read_csv(f/f'stab_nogueira_seed{s}.csv') for s in seeds]).to_csv(f/'stab_nogueira.csv',index=False)
        pd.DataFrame([j['correlations'] for j in js]).to_csv(f/'score_correlations.csv',index=False)
        pd.DataFrame([j['stratified'] for j in js]).to_csv(f/'stab_stratified.csv',index=False)
        pd.DataFrame([row for j in js for row in j['overlaps']]).to_csv(f/'overlap_pca_nisp.csv',index=False)
        if tag=='fmnist_bal':pd.DataFrame([row for j in js for row in j['mixture']]).to_csv(f/'mixture_shift.csv',index=False)
        if tag in CROSS:spec.extend([dict(tag=tag,**j['spectral']) for j in js])
        weight.extend([dict(tag=tag,seed=seed,rho=j['weight_rho'],inf_fs_cv=j['inf_fs_cv']) for seed,j in zip(seeds,js)])
    pd.DataFrame(spec).to_csv(R/'spectral_check.csv',index=False);pd.DataFrame(weight).to_csv(R/'nisp_weight_diagnostic.csv',index=False)

if __name__=='__main__':
    diagnostics()
