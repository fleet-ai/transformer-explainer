#!/usr/bin/env python3
"""Record all 38 pretrained Zamba2 blocks and the tied vocabulary projection.

Uses the installed official Transformers modules, including all six adapters.
Only CPU execution chunk size changes (256 -> 16), to bound SSD working memory.
All activations are stored losslessly; state samples retain every head and all
128 state coordinates for the first four channels, explicitly labeled as such.
"""
from __future__ import annotations
import argparse, gc, hashlib, inspect, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer
from transformers import Zamba2Config
from transformers.models.zamba2 import modeling_zamba2 as zamba
from capture_mla_hybrid_common import Checkpoint, digest

ROOT=Path(__file__).resolve().parents[1]
MODEL='Zyphra/Zamba2-1.2B'
REVISION='6b05bf29d1bb4ca71a36d12f7da4d3120dcde7fe'

class BoundedCheckpoint(Checkpoint):
    def __init__(self, cache, reuse):
        self.reuse=Path(reuse)/MODEL.replace('/','--')/REVISION
        super().__init__(MODEL,REVISION,'model.safetensors',cache)

    def fetch(self,file,start=None,end=None):
        suffix='' if start is None else f'.{start}-{end}'
        p=self.cache/(file.replace('/','__')+suffix)
        old=self.reuse/p.name
        expected=None if start is None else end-start+1
        if not p.exists() and old.exists() and (expected is None or old.stat().st_size==expected):
            os.link(old,p)
        if not p.exists():
            if expected is not None and expected>8*1024*1024:
                chunks=[(i,min(end,i+8*1024*1024-1)) for i in range(start,end+1,8*1024*1024)]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    pieces=list(pool.map(lambda ab:self.fetch(file,*ab),chunks))
                tmp=p.with_name(p.name+'.partial');tmp.write_bytes(b''.join(pieces));tmp.rename(p)
            else:
                cmd=['curl','--http1.1','--fail','--location','--silent','--show-error','--retry','5',
                    '--retry-all-errors','--connect-timeout','20','--max-time','180']
                if expected is not None:cmd+=['--range',f'{start}-{end}','--max-filesize',str(max(16384,expected+1024))]
                tmp=p.with_name(p.name+'.partial')
                subprocess.run(cmd+[self.url+file+('?full_trace_range='+suffix if suffix else ''),'-o',str(tmp)],check=True)
                if expected is not None:assert tmp.stat().st_size==expected,p.name
                tmp.rename(p)
        data=p.read_bytes()
        if expected is not None:assert len(data)==expected,p.name
        return data

def save_tensors(path,tensors):
    path.parent.mkdir(parents=True,exist_ok=True);offset=0;entries={};sha=hashlib.sha256()
    with path.open('wb') as f:
        for name,value in tensors.items():
            a=value.detach().cpu().contiguous().numpy().astype('<f4',copy=False)
            assert np.isfinite(a).all(),name
            data=a.tobytes();f.write(data);sha.update(data)
            entries[name]={'offset':offset,'length':a.size,'shape':list(a.shape)};offset+=a.size
    return {'file':path.name,'sha256':sha.hexdigest(),'bytes':path.stat().st_size,'tensors':entries}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache-dir',default='/private/tmp/moe-explainer-hybrid-full-cache')
    parser.add_argument('--reuse-dir',default='/private/tmp/moe-architecture-cache')
    args=parser.parse_args();torch.set_num_threads(2);torch.set_grad_enabled(False)
    cp=BoundedCheckpoint(args.cache_dir,args.reuse_dir)
    raw=json.loads(cp.asset('config.json').read_text())
    config=Zamba2Config(**{**raw,'chunk_size':16});config._attn_implementation='eager'
    assert len(config.layers_block_type)==38 and config.tie_word_embeddings
    tokenizer=Tokenizer.from_file(str(cp.asset('tokenizer.json')))
    prompt='The capital of France is';ids=tokenizer.encode(prompt).ids
    tokens=[tokenizer.decode([i],skip_special_tokens=False) for i in ids];n=len(ids)
    embedding=torch.stack([cp.tensor('model.embed_tokens.weight',row=i) for i in ids])[None]
    x=embedding.clone();reference=x.clone()
    pos=torch.arange(n)[None]
    cos,sin=zamba.Zamba2RotaryEmbedding(config)(x,pos)
    mask=torch.triu(torch.full((n,n),torch.finfo(torch.float32).min),diagonal=1)[None,None]
    root=ROOT/'static/traces/hybrid';root.mkdir(parents=True,exist_ok=True)
    checks={'hookedVsUnhookedMaxError':0.,'stateScanMaxError':0.,'scanProjectionMaxError':0.,
        'residualMaxError':0.,'attentionMaxError':0.,'attentionContextMaxError':0.,'feedForwardMaxError':0.}
    layers=[];shared=None
    for index,kind in enumerate(config.layers_block_type):
        print(f'Loading Zamba2 block {index+1}/38 ({kind})',flush=True)
        prefix=f'model.layers.{index}.'
        names=[k for k in cp.header if k.startswith(prefix)]
        cp.prefetch(names)
        if kind=='hybrid' and shared is None:
            with torch.device('meta'):shared=zamba.Zamba2AttentionDecoderLayer(config,block_id=0)
            shared_prefix='model.layers.5.shared_transformer.'
            shared.load_state_dict({k.removeprefix(shared_prefix):cp.tensor(k) for k in names if k.startswith(shared_prefix)},strict=True,assign=True)
            shared.eval()
        with torch.device('meta'):
            mamba=zamba.Zamba2MambaDecoderLayer(config,layer_idx=index)
            linear=nn.Linear(2048,2048,bias=False) if kind=='hybrid' else None
        mp=prefix+('mamba_decoder.' if kind=='hybrid' else '')
        mamba.load_state_dict({k.removeprefix(mp):cp.tensor(k) for k in names if k.startswith(mp)},strict=True,assign=True)
        if linear is not None:linear.load_state_dict({'weight':cp.tensor(prefix+'linear.weight')},strict=True,assign=True)
        module=zamba.Zamba2HybridLayer(shared,linear,mamba) if kind=='hybrid' else mamba
        module.eval();mixer=mamba.mamba
        kwargs={'original_hidden_states':embedding,'layer_idx':index,'causal_mask':mask,
            'output_attentions':True,'position_embeddings':(cos,sin)}
        reference=module(reference,**kwargs)[0]
        recorded={};handles=[]
        def hook(mod,name):
            def record(mod,args,out):
                recorded[name]=(args, out)
            handles.append(mod.register_forward_hook(record))
        for name,mod in [('norm1',mamba.input_layernorm),('projection',mixer.in_proj),('conv',mixer.conv1d),
            ('gated',mixer.norm),('mambaOutput',mixer)]:hook(mod,name)
        if kind=='hybrid':
            for name,mod in [('attentionInput',shared.input_layernorm),('attention',shared.self_attn),
                ('ffnInput',shared.pre_ff_layernorm),('down',shared.feed_forward),('transformerOutput',linear)]:hook(mod,name)
        out=module(x,**kwargs)[0]
        for handle in handles:handle.remove()
        error=float((out-reference).abs().max());assert error==0
        checks['hookedVsUnhookedMaxError']=max(checks['hookedVsUnhookedMaxError'],error)
        r=lambda key:recorded[key][1][0]
        projection=r('projection');gate,xbc,dt_raw=projection.split([4096,4352,64],dim=-1)
        conv_raw=r('conv')[:,:n].transpose(0,1)
        conv=torch.nn.functional.silu(conv_raw)
        xx,B,C=conv.split([4096,128,128],dim=-1);xx=xx.reshape(n,64,64)
        dt=torch.nn.functional.softplus(dt_raw+mixer.dt_bias).clamp(min=mixer.time_step_min)
        A=-torch.exp(mixer.A_log.float());D=mixer.D
        state=torch.zeros(64,64,128);states=[];ys=[]
        for token in range(n):
            decay=torch.exp(dt[token]*A)
            inject=dt[token,:,None,None]*B[token,None,None,:]*xx[token,:,:,None]
            state=state*decay[:,None,None]+inject
            states.append(state[:,:4].clone())
            ys.append((state*C[token,None,None,:]).sum(-1)+D[:,None]*xx[token])
        sequential=torch.stack(ys).reshape(n,4096)
        scan=recorded['gated'][0][0][0]
        scan_error=float((sequential-scan).abs().max())
        assert torch.allclose(sequential,scan,atol=5e-4,rtol=2e-4),(index,scan_error)
        checks['stateScanMaxError']=max(checks['stateScanMaxError'],scan_error)
        output_check=mixer.out_proj(mixer.norm(sequential[None],gate[None]))[0]
        output_error=float((output_check-r('mambaOutput')).abs().max())
        assert torch.allclose(output_check,r('mambaOutput'),atol=2e-4,rtol=2e-4),(index,output_error)
        checks['scanProjectionMaxError']=max(checks['scanProjectionMaxError'],output_error)
        assert torch.equal(x[0]+r('mambaOutput'),out[0])
        data={'input':x[0],'norm1':r('norm1'),'mambaProjection':projection,'gate':gate,'convInput':xbc,
            'convRaw':conv_raw,'convOutput':conv,'x':xx,'B':B,'C':C,'dtRaw':dt_raw,'dt':dt,'dtBias':mixer.dt_bias,
            'A':A,'D':D,'state':torch.stack(states),'scanOutput':scan,'sequentialReadout':sequential,
            'gated':r('gated'),'mambaOutput':r('mambaOutput'),'output':out[0],
            'inWeights':mixer.in_proj.weight[:4],'outWeights':mixer.out_proj.weight[:4],
            'convWeights':mixer.conv1d.weight,'convBias':mixer.conv1d.bias,
            'normWeight':mamba.input_layernorm.weight,'gatedNormWeight':mixer.norm.weight}
        if kind=='hybrid':
            adapter=config.hybrid_layer_ids.index(index);attn=shared.self_attn;inp=r('attentionInput')
            def proj(name):
                return (getattr(attn,name+'_proj')(inp)+getattr(attn,'linear_'+name+'_adapter_list')[adapter](inp)).reshape(n,32,128).transpose(0,1)
            q0,k0,v=proj('q'),proj('k'),proj('v')
            q,k=zamba.apply_rotary_pos_emb(q0[None],k0[None],cos,sin);q,k=q[0],k[0]
            attn_output,weights=recorded['attention'][1];weights=weights[0]
            scores=q@k.transpose(-1,-2);rebuilt=(scores*attn.scaling+mask[0]).softmax(-1)
            err=float((rebuilt-weights).abs().max());assert err<1e-6
            checks['attentionMaxError']=max(checks['attentionMaxError'],err)
            context=weights@v
            err=float((attn.o_proj(context.transpose(0,1).reshape(n,4096))-attn_output[0]).abs().max());assert err<1e-5
            checks['attentionContextMaxError']=max(checks['attentionContextMaxError'],err)
            f=shared.feed_forward;finput=r('ffnInput')
            gu=f.gate_up_proj(finput)+f.gate_up_proj_adapter_list[adapter](finput)
            g,u=gu.chunk(2,-1);gelu=f.act_fn(g);product=gelu*u
            err=float((f.down_proj(product)-r('down')).abs().max());assert err<1e-5
            checks['feedForwardMaxError']=max(checks['feedForwardMaxError'],err)
            effective=f.gate_up_proj.weight[:4]+f.gate_up_proj_adapter_list[adapter][1].weight[:4]@f.gate_up_proj_adapter_list[adapter][0].weight
            effective_up=f.gate_up_proj.weight[8192:8196]+f.gate_up_proj_adapter_list[adapter][1].weight[8192:8196]@f.gate_up_proj_adapter_list[adapter][0].weight
            data.update({'query':q,'key':k,'value':v,'queryBefore':q0,'keyBefore':k0,'scores':scores,
                'attention':weights,'context':context,'attentionInput':inp,'attentionOutput':attn_output[0],
                'ffnInput':finput,'gateFFN':g,'up':u,'gelu':gelu,'product':product,'down':r('down'),
                'transformerOutput':r('transformerOutput'),'gateWeights':effective,'upWeights':effective_up,
                'downWeights':f.down_proj.weight[:4],'transformerWeights':linear.weight[:4]})
        entry=save_tensors(root/f'block{index:02}.f32',data)
        layers.append({'index':index,'type':kind,**entry})
        print(f'Captured block {index+1}/38: {entry["bytes"]:,} bytes, state scan error {scan_error:.3g}',flush=True)
        x=out.detach();del module,mamba,mixer,data,recorded,out;gc.collect()
    norm=zamba.Zamba2RMSNorm(2048,eps=config.rms_norm_eps)
    norm.load_state_dict({'weight':cp.tensor('model.final_layernorm.weight')})
    final=norm(x)[0];reference_final=norm(reference)[0]
    # The checkpoint ties its vocabulary projection to the complete embedding table.
    weights=cp.tensor('model.embed_tokens.weight')
    logits=torch.nn.functional.linear(final[-1],weights)
    reference_logits=torch.nn.functional.linear(reference_final[-1],weights)
    assert torch.equal(logits,reference_logits)
    top=logits.topk(4).indices;pred=int(logits.argmax())
    common=save_tensors(root/'model.f32',{'embedding':embedding[0],'finalHidden':final,'logits':logits,
        'cos':cos[0],'sin':sin[0],'classifierWeights':weights[top],'classifierTokenIds':top.float(),
        'finalNormWeight':norm.weight})
    provenance=cp.provenance('Complete pretrained embedding, all 38 blocks (32 Mamba2 and six shared Transformer + Mamba2 blocks with distinct adapters), final RMSNorm and all 32,000 tied LM logits.',__file__,digest(Path(inspect.getfile(zamba)).read_bytes()))
    provenance['partial']=False
    model={'architecture':'hybrid','id':MODEL,'name':'Zamba2-1.2B','revision':REVISION,'url':provenance['modelUrl'],
        'license':'apache-2.0','numLayers':38,'hiddenSize':2048,'intermediateSize':8192,'numExperts':0,'topK':0,
        'attentionHeads':32,'keyValueHeads':32,'attentionHeadSize':128,'valueHeadSize':128,'attentionMultiplier':.125,
        'embeddingMultiplier':1,'residualMultiplier':1,'logitsScaling':1,'ropeTheta':10000,'vocabSize':32000,
        'normalization':'rmsnorm','rmsNormEps':1e-5,'maskMode':'causal','outputKind':'language','activation':'gelu',
        'mambaHeads':64,'mambaHeadSize':64,'stateSize':128,'convolutionWidth':4,'stateChannelsRecorded':4,
        'layerTypes':config.layers_block_type,'hybridLayers':config.hybrid_layer_ids,
        'executionChunkSize':16,'checkpointChunkSize':256}
    # A single-token SentencePiece decode strips its initial word boundary.
    # Preserve that trained token boundary when appending a vocabulary candidate.
    vocabulary=[]
    for token_id in range(32000):
        decoded=tokenizer.decode([token_id],skip_special_tokens=False)
        if (tokenizer.id_to_token(token_id) or '').startswith('▁') and not decoded.startswith(' '):decoded=' '+decoded
        vocabulary.append(decoded)
    doc={'schemaVersion':1,'kind':'hybrid-full','model':model,'prompt':prompt,'tokens':tokens,'tokenIds':ids,
        'sequenceLength':n,'vocabulary':vocabulary,
        'common':common,'layers':layers,'nextTokenId':pred,'nextToken':vocabulary[pred],
        'validation':checks,'capture':provenance}
    (root/'manifest.json').write_text(json.dumps(doc,separators=(',',':'),allow_nan=False)+'\n')
    print('Complete Zamba2 output',repr(doc['nextToken']),checks,flush=True)

if __name__=='__main__':main()
