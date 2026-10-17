"""Bounded public safetensors reads for pretrained architecture-prefix traces.

No checkpoint executable code is run by this downloader. Every HTTP range is
size checked, cached outside the repository, and SHA-256 recorded in provenance.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import struct
import subprocess
import torch

DTYPES = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16}


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Checkpoint:
    def __init__(self, model, revision, filename, cache):
        self.model, self.revision, self.filename = model, revision, filename
        self.cache = Path(cache) / model.replace('/', '--') / revision
        self.cache.mkdir(parents=True, exist_ok=True)
        self.url = f'https://huggingface.co/{model}/resolve/{revision}/'
        self.manifest = {}
        self.assets = {}
        first = self.fetch(filename, start=0, end=7)
        self.header_size = struct.unpack('<Q', first)[0]
        self.header_bytes = self.fetch(filename, start=8, end=7 + self.header_size)
        self.header = json.loads(self.header_bytes)
        self.offset = self.header_size + 8

    def fetch(self, file, start=None, end=None):
        suffix = '' if start is None else f'.{start}-{end}'
        p = self.cache / (file.replace('/', '__') + suffix)
        if not p.exists():
            cmd = ['curl', '--fail', '--location', '--silent', '--show-error', '--retry', '3']
            if start is not None:
                cmd += ['--range', f'{start}-{end}', '--max-filesize', str(max(16384,end-start+1024))]
            # Range is also part of the URL cache key; intermediary caches can
            # otherwise return a prior byte range for the same resource URL.
            url = self.url + file + ('?trace_range=' + suffix if suffix else '')
            tmp = p.with_name(p.name + '.partial')
            subprocess.run(cmd + [url, '-o', str(tmp)], check=True)
            if start is not None and tmp.stat().st_size != end-start+1:
                raise ValueError(f'Wrong HTTP range size for {file} {suffix}')
            tmp.rename(p)
        data = p.read_bytes()
        if start is not None and len(data) != end-start+1:
            raise ValueError('Cached range has wrong size')
        return data

    def tensor(self, name, row=None):
        spec = self.header[name]
        a,b = spec['data_offsets']
        shape = spec['shape']
        if row is not None:
            stride = (b-a)//shape[0]
            a += row*stride
            b = a+stride
            shape = shape[1:]
        data = self.fetch(self.filename, self.offset+a, self.offset+b-1)
        label = name if row is None else f'{name}[{row}]'
        self.manifest[label] = {'shape':shape, 'dtype':spec['dtype'], 'sha256':digest(data), 'byteRange':[self.offset+a,self.offset+b-1]}
        return torch.frombuffer(bytearray(data), dtype=DTYPES[spec['dtype']]).clone().reshape(shape).float()

    def prefetch(self, names):
        def read(name):
            s=self.header[name]; a,b=s['data_offsets']; self.fetch(self.filename,self.offset+a,self.offset+b-1)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(read,names))

    def asset(self, file):
        data=self.fetch(file)
        self.assets[file]=digest(data)
        return self.cache / file

    def provenance(self, scope, script, source_hash=None):
        import transformers
        return {'modelId':self.model,'revision':self.revision,'trained':True,'partial':True,
                'scope':scope,'dtype':'float32 computation from checkpoint tensors',
                'captureScript':'scripts/'+Path(script).name,'modelUrl':f'https://huggingface.co/{self.model}/tree/{self.revision}','captureScriptSha256':digest(Path(script).read_bytes()),
                'libraryVersions':{'torch':torch.__version__,'transformers':transformers.__version__},
                'weightsSha256':self.manifest,'assetsSha256':self.assets,'captureHelperSha256':digest(Path(__file__).read_bytes()),'headerSha256':digest(self.header_bytes),
                **({'modelSourceSha256':source_hash} if source_hash else {})}


def tensor(name, data, tokens=None, shape=None, slice=None, **extra):
    t=data.detach().cpu().float()
    result={'name':name, 'shape':list(shape or t.shape), 'values':t.tolist(), **extra}
    if tokens: result['rowLabels']=tokens
    if slice: result['slice']=slice
    return result


def sample(name, data, tokens, width=4, **extra):
    t=data.detach().float()
    if t.ndim==3 and t.shape[0]==1: t=t[0]
    return tensor(name,t[:,:width],tokens,shape=t.shape,slice=f'All {len(tokens)} token rows; coordinates 0–{width-1} of {t.shape[-1]}.',**extra)


def operation(id,label,kind,description,formula,tensors,metrics=None):
    d={'id':id,'label':label,'kind':kind,'description':description,'formula':formula,'tensors':tensors}
    if metrics: d['metrics']=metrics
    return d


def write_trace(path,data):
    Path(path).write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
    print(f'Wrote {path}: {Path(path).stat().st_size:,} bytes',flush=True)
