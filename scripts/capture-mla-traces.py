#!/usr/bin/env python3
"""Record DeepSeek-V2-Lite's first pretrained MLA layer without downloading its MoE.

Runs the pinned upstream attention, normalization, and rotary classes, extracted
verbatim from the reviewed Apache-2.0 model source. The complete source hash is
checked before execution. Downloads only the six required tensors and embedding
rows. This is an attention-prefix trace, never a full language-model prediction.
"""
import argparse
import ast
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple
import warnings
import torch
from torch import nn
from tokenizers import Tokenizer
from transformers.cache_utils import Cache
from capture_mla_hybrid_common import Checkpoint, digest, operation, sample, tensor, write_trace

MODEL='deepseek-ai/DeepSeek-V2-Lite'
REVISION='604d5664dddd88a0433dbae533b7fe9472482de0'


def load_source(path):
    source=path.read_bytes()
    assert digest(source)=='7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c', 'Pinned model source changed'
    # Only these reviewed mathematical definitions are executed; no model
    # repository imports, top-level calls, loading hooks, or tokenizer code.
    names={'DeepseekV2RMSNorm','DeepseekV2RotaryEmbedding','DeepseekV2LinearScalingRotaryEmbedding',
           'DeepseekV2DynamicNTKScalingRotaryEmbedding','DeepseekV2YarnRotaryEmbedding',
           'yarn_find_correction_dim','yarn_find_correction_range','yarn_get_mscale','yarn_linear_ramp_mask',
           'rotate_half','apply_rotary_pos_emb','DeepseekV2Attention'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes)==len(names)
    env={'torch':torch,'nn':nn,'math':math,'warnings':warnings,'Optional':Optional,'Tuple':Tuple,
         'Cache':Cache,'DeepseekV2Config':SimpleNamespace,'logger':logging.getLogger('mla')}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env,digest(source)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache-dir',default='/private/tmp/moe-architecture-cache');parser.add_argument('--output',default='static/architectures/mla.json');args=parser.parse_args()
    torch.set_num_threads(4)
    cp=Checkpoint(MODEL,REVISION,'model-00001-of-000004.safetensors',args.cache_dir)
    config=json.loads(cp.asset('config.json').read_text())
    source=cp.asset('modeling_deepseek.py')
    env,source_hash=load_source(source)
    tokenizer=Tokenizer.from_file(str(cp.asset('tokenizer.json')))
    tokenizer_config=json.loads(cp.asset('tokenizer_config.json').read_text())
    prefix='model.layers.0.'
    names=[n for n in cp.header if n.startswith(prefix+'self_attn.') or n==prefix+'input_layernorm.weight']
    print('MLA tensor bytes',sum(cp.header[n]['data_offsets'][1]-cp.header[n]['data_offsets'][0] for n in names),flush=True)
    cp.prefetch(names)
    with torch.no_grad():
        attn=env['DeepseekV2Attention'](SimpleNamespace(**config),layer_idx=0).eval()
        attn.load_state_dict({n.removeprefix(prefix+'self_attn.'):cp.tensor(n) for n in names if n.startswith(prefix+'self_attn.')},strict=True)
        norm=env['DeepseekV2RMSNorm'](config['hidden_size'],config['rms_norm_eps']).eval()
        norm.load_state_dict({'weight':cp.tensor(prefix+'input_layernorm.weight')})
        examples=[]
        for eid,text in [('france','The capital of France is'),('code','def add(a, b): return a + b')]:
            ids=tokenizer.encode(text).ids
            if tokenizer_config.get('add_bos_token') and ids[0]!=config['bos_token_id']:
                ids.insert(0,config['bos_token_id'])
            tokens=['<BOS>' if i==config['bos_token_id'] else tokenizer.decode([i],skip_special_tokens=False) for i in ids]
            x=torch.stack([cp.tensor('model.embed_tokens.weight',row=i) for i in ids])[None]
            hidden=norm(x); n=len(ids); pos=torch.arange(n)[None]
            mask=torch.triu(torch.full((n,n),torch.finfo(torch.float32).min),diagonal=1)[None,None]
            y,weights,_=attn(hidden,attention_mask=mask,position_ids=pos,output_attentions=True)
            # Recover intermediates using the exact module calls used by forward;
            # independently reconstruct and compare against recorded output.
            q=attn.q_proj(hidden).view(1,n,16,192).transpose(1,2)
            q_content,q_rot=q.split([128,64],dim=-1)
            compressed=attn.kv_a_proj_with_mqa(hidden)
            latent_raw,k_rot=compressed.split([512,64],dim=-1)
            latent=attn.kv_a_layernorm(latent_raw)
            expanded=attn.kv_b_proj(latent).view(1,n,16,256).transpose(1,2)
            k_content,v=expanded.split([128,128],dim=-1)
            k_rot=k_rot.view(1,n,1,64).transpose(1,2)
            cos,sin=attn.rotary_emb(v,seq_len=n)
            qr,kr=env['apply_rotary_pos_emb'](q_rot,k_rot,cos,sin,pos)
            query=torch.cat([q_content,qr],dim=-1)
            key=torch.cat([k_content,kr.expand(1,16,n,64)],dim=-1)
            scores=(query@key.transpose(-1,-2))*attn.softmax_scale
            rebuilt_weights=torch.softmax(scores+mask,dim=-1)
            heads=rebuilt_weights@v
            rebuilt=attn.o_proj(heads.transpose(1,2).reshape(1,n,2048))
            err=float((rebuilt-y).abs().max())
            assert torch.allclose(rebuilt,y,atol=1e-5,rtol=1e-5)
            assert torch.allclose(rebuilt_weights,weights,atol=1e-6,rtol=1e-6)
            def op(*a,**kw):return operation(*a,**kw)
            operations=[
              op('embedding','Token embedding','embedding','The selected prompt uses actual pretrained embedding rows, then the first layer’s RMSNorm.','x = RMSNorm(E[token])',[sample('Embedding',x,tokens),sample('Normalized input x',hidden,tokens)], [{'label':'Hidden width','value':2048}]),
              op('query','Query projection','projection','This Lite checkpoint projects directly from 2048D to 16 query heads. Each 192D head is split into 128 content coordinates and 64 rotary coordinates; there is no query low-rank bottleneck in this checkpoint.','Q = WQ x; qₕ = [qₕᶜ(128), qₕᴿ(64)]',[sample('Query projection (all heads)',q.transpose(1,2).reshape(1,n,3072),tokens),sample('Query head 1',q[0,0],tokens)],[{'label':'Query projection width','value':3072},{'label':'Query heads','value':16},{'label':'Per-head width','value':192}]),
              op('compress','Compress K/V','projection','A shared projection creates a 512D latent and a separate 64D positional key. The normalized latent can be cached rather than full per-head K and V.','[c_raw, kᴿ] = Wᴰᴷⱽ x; c = RMSNorm(c_raw)',[sample('Raw latent c',latent_raw,tokens),sample('Normalized latent c',latent,tokens),sample('Shared positional key before RoPE',k_rot[0,0],tokens)], [{'label':'Latent coordinates','value':512},{'label':'Positional key coordinates','value':64},{'label':'Latent + positional cache per token','value':576},{'label':'Expanded K + V per token','value':5120}]),
              op('rope','Decoupled RoPE','rope','Each head has 128 content-query coordinates plus 64 rotary coordinates. A single rotated key is shared by all 16 heads. This checkpoint uses its trained YaRN configuration.','q = [qᶜ, RoPE(qᴿ)]; kₕ = [kₕᶜ, RoPE(kᴿ)]',[sample('Q head 1 content',q_content[0,0],tokens),sample('Q head 1 rotary before',q_rot[0,0],tokens),sample('Q head 1 rotary after',qr[0,0],tokens),sample('Shared key rotary after',kr[0,0],tokens),sample('cos θ',cos,tokens),sample('sin θ',sin,tokens)], [{'label':'Q/K head width','value':192},{'label':'Rotary width','value':64}]),
              op('expand','Expand latent to heads','projection','For this reference forward pass, the shared 512D latent expands into 16 content-key and value heads. Optimized inference may absorb these linear maps into surrounding projections.','[kₕᶜ, vₕ] = Wᵁᴷⱽₕ c',[sample('K head 1 content',k_content[0,0],tokens),sample('V head 1',v[0,0],tokens)], [{'label':'Heads','value':16},{'label':'Content key width','value':128},{'label':'Value width','value':128},{'label':'Expanded projection width','value':4096}]),
              op('attention','Causal latent attention','attention','Attention uses content and positional dot products together. Each matrix is recorded from the actual first-layer forward pass; future positions are masked.','Aₕ = softmax((qₕ kₕᵀ) × scale + causal mask); oₕ = Aₕ vₕ',[
                  tensor(f'Head {h+1} attention',weights[0,h],tokens,columnLabels=tokens,role='attention',head=h) for h in range(16)
              ]+[sample('Head 1 attended values',heads[0,0],tokens)], [{'label':'Checkpoint attention scale','value':attn.softmax_scale}]),
              op('output','Attention output','output','The 16 value-head outputs are concatenated and projected back to 2048D. This trace stops before the feed-forward layer; it is not a full-model next-token prediction.','y = Wᴼ concat(o₁,…,o₁₆); residual = embedding + y',[sample('Attention output y',y,tokens),sample('Residual after attention',x+y,tokens)], [{'label':'Output width','value':2048},{'label':'Independent reconstruction max error','value':err}])
            ]
            examples.append({'id':eid,'label':text,'input':{'text':text},'tokens':tokens,'tokenIds':ids,'verification':[{'name':'Output projection reconstruction','maxAbsoluteError':err,'tolerance':1e-5},{'name':'Attention reconstruction','maxAbsoluteError':float((rebuilt_weights-weights).abs().max()),'tolerance':1e-6},{'name':'Causal mask','maxAbsoluteError':float(weights.triu(1).abs().max()),'tolerance':0.0},{'name':'Attention row normalization','maxAbsoluteError':float((weights.sum(-1)-1).abs().max()),'tolerance':1e-6}],'layers':[{'index':0,'label':'Pretrained MLA layer 1','operations':operations}]})
        result={'schemaVersion':1,'architecture':'mla','title':'DeepSeek-style latent attention',
          'provenance':cp.provenance('DeepSeek-V2-Lite pretrained embedding and first MLA attention layer only. No feed-forward, later layers, LM head, or generated prediction.',__file__,source_hash),
          'config':{'hiddenSize':2048,'attentionHeads':16,'headDim':192,'contentKeyDim':128,'rotaryDim':64,'valueDim':128,'kvLatentDim':512,'layers':27,'capturedLayers':1,'cacheCoordinatesPerToken':576,'expandedKVCoordinatesPerToken':5120},'examples':examples}
        write_trace(args.output,result)

if __name__=='__main__':main()
