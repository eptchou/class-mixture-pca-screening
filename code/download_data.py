from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
import hashlib,json,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parent
import os
RAW=Path(os.environ.get('ADAC_DATA_DIR', str(ROOT/'data')))/'raw';RAW.mkdir(parents=True,exist_ok=True)
expected={v['file']:v['sha256'] for v in json.loads((ROOT/'provenance/download_manifest.json').read_text())}
urls={}
base='https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
for n in ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']:
    urls[n]=base+n
for n in ['mfeat_factors','mfeat_fourier','mfeat_karhunen','mfeat_morphological','mfeat_pixel','mfeat_zernike']:
    urls[n+'.tsv.gz']='https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/'+n+'/'+n+'.tsv.gz'
urls['reuters.zip']='https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/reuters.zip'

def get(item):
    name,url=item;p=RAW/name
    if not p.exists():
        tmp=p.with_suffix(p.suffix+'.part')
        with urlopen(url,timeout=45) as r,tmp.open('wb') as f:shutil.copyfileobj(r,f)
        tmp.replace(p)
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    if digest!=expected[name]:raise RuntimeError(f'Checksum mismatch: {name}')
    print(name,p.stat().st_size,flush=True)
    return {'file':name,'url':url,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

manifest=list(ThreadPoolExecutor(5).map(get,urls.items()))
(RAW.parent/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
subprocess.run([sys.executable,str(ROOT/'prep_data.py')],check=True)
