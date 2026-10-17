#!/usr/bin/env python3
"""Capture the actual first six pretrained Zamba2-1.2B layers on CPU.

The first five blocks are Mamba2; block six includes the shared Transformer
followed by Mamba2. Only needed checkpoint tensors and prompt embedding rows
are downloaded. This stops before the remaining 32 blocks and LM head.
"""
import argparse
import gc
import inspect
import json
from pathlib import Path
import re
import torch
from torch import nn
from tokenizers import Tokenizer
from transformers import Zamba2Config
from transformers.models.zamba2 import modeling_zamba2 as zamba
from capture_mla_hybrid_common import Checkpoint,digest,operation,sample,tensor,write_trace

MODEL='Zyphra/Zamba2-1.2B'
REVISION='6b05bf29d1bb4ca71a36d12f7da4d3120dcde7fe'


def capture_mamba(mixer, cap, tokens):
    """Independent sequential state recurrence compared with library SSD scan."""
    n=len(tokens);p=cap['mamba_projection'];gate,xbc,dt_raw=p.split([4096,4352,64],dim=-1)
    conv=torch.nn.functional.silu(cap['conv_raw'][:,:,:n].transpose(1,2))
    x,B,C=conv.split([4096,128,128],dim=-1)
    x=x.reshape(1,n,64,64);B=B.reshape(1,n,1,128).expand(-1,-1,64,-1);C=C.reshape(1,n,1,128).expand(-1,-1,64,-1)
    dt=torch.nn.functional.softplus(dt_raw+mixer.dt_bias).clamp(min=mixer.time_step_min)
    A=-torch.exp(mixer.A_log.float());D=mixer.D
    state=torch.zeros(1,64,64,128)
    previous=[];states=[];decays=[];injections=[];ys=[]
    for t in range(n):
        previous.append(state[0,0,0,:4].clone())
        decay=torch.exp(dt[:,t]*A)
        inject=dt[:,t,:,None,None]*B[:,t,:,None,:]*x[:,t,:,:,None]
        state=state*decay[:,:,None,None]+inject
        yt=(state*C[:,t,:,None,:]).sum(-1)+D[None,:,None]*x[:,t]
        states.append(state[0,0,0,:4].clone());decays.append(decay[0,0].repeat(4));injections.append(inject[0,0,0,:4].clone());ys.append(yt.reshape(1,4096))
    y=torch.stack(ys,dim=1)
    err=float((y-cap['scan_output']).abs().max())
    # The installed implementation computes mathematically identical chunked
    # products with different float32 accumulation order.
    assert torch.allclose(y,cap['scan_output'],atol=5e-4,rtol=2e-4),err
    projected=mixer.out_proj(mixer.norm(y,gate))
    output_err=float((projected-cap['mamba_output']).abs().max())
    assert torch.allclose(projected,cap['mamba_output'],atol=2e-4,rtol=2e-4),output_err
    prev=torch.stack(previous);new=torch.stack(states);decay=torch.stack(decays);inject=torch.stack(injections)
    recurrence_error=float((prev*decay+inject-new).abs().max());assert recurrence_error==0
    ops=[operation('conv','Causal convolution','convolution','A learned width-4 depthwise causal convolution mixes recent projected inputs. SiLU produces the input-dependent x, B, and C used by the selective state-space scan.','[z, xBC, Δ_raw] = W_in RMSNorm(h); [x, B, C] = SiLU(CausalConv₄(xBC))',[
        sample('Normalized Mamba input',cap['mamba_input'],tokens),sample('Convolution input xBC',xbc,tokens),sample('Convolution + SiLU x',x.reshape(1,n,4096),tokens),sample('Input-dependent B',B[0,:,0],tokens),sample('Input-dependent C',C[0,:,0],tokens)], [{'label':'Convolution width','value':4},{'label':'Inner width','value':4096}]),
        operation('selective-state','Selective state update','state-space','The state carries information forward without a token-by-token attention matrix. The four saved state cells are head 1, channel 0, state coordinates 0–3. Their decay and injection are recorded for every token.','Δ_t = softplus(Δ_raw + bias); S_t = exp(Δ_t A) S_(t−1) + Δ_t B_t x_t; y_t = C_t S_t + D x_t',[
          sample('Time step Δ (heads 1–4)',dt,tokens),
          tensor('Previous state S_(t−1)',prev,tokens,shape=[n,64,64,128],slice='Head 1, channel 0, state coordinates 0–3.'),
          tensor('Decay exp(Δ A)',decay,tokens,shape=[n,64],slice='Head 1 decay repeated across its four displayed state coordinates.'),
          tensor('Input injection Δ B x',inject,tokens,shape=[n,64,64,128],slice='Head 1, channel 0, state coordinates 0–3.'),
          tensor('Updated state S_t',new,tokens,shape=[n,64,64,128],slice='Head 1, channel 0, state coordinates 0–3.'),
          sample('State readout + skip y',cap['scan_output'],tokens),sample('Gated RMSNorm output',cap['gated'],tokens)],
          [{'label':'SSM heads','value':64},{'label':'Channels per head','value':64},{'label':'State coordinates per channel','value':128},{'label':'State scalars per Mamba layer','value':64*64*128},{'label':'Sequential scan max difference','value':err}])]
    checks=[{'name':'Sequential recurrence vs library SSD scan','maxAbsoluteError':err,'tolerance':5e-4},
            {'name':'Reconstructed Mamba output','maxAbsoluteError':output_err,'tolerance':2e-4},
            {'name':'Displayed state-cell recurrence','maxAbsoluteError':recurrence_error,'tolerance':0}]
    return ops,checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache-dir',default='/private/tmp/moe-architecture-cache');parser.add_argument('--output',default='static/architectures/hybrid.json');args=parser.parse_args()
    torch.set_num_threads(4)
    cp=Checkpoint(MODEL,REVISION,'model.safetensors',args.cache_dir)
    raw=json.loads(cp.asset('config.json').read_text())
    # Only instantiate the prefix and its first adapter. Execution chunk size
    # changes memory use, not the recurrence, parameters, or architecture.
    prefix_config={**raw,'num_hidden_layers':6,'layers_block_type':raw['layers_block_type'][:6],'chunk_size':16}
    config=Zamba2Config(**prefix_config);config._attn_implementation='eager'
    names=[k for k in cp.header if any(k.startswith(f'model.layers.{i}.') for i in range(6)) and not re.search(r'adapter_list\.[1-5]\.',k)]
    print('Hybrid prefix tensor bytes',sum(cp.header[n]['data_offsets'][1]-cp.header[n]['data_offsets'][0] for n in names),flush=True)
    cp.prefetch(names)
    tokenizer=Tokenizer.from_file(str(cp.asset('tokenizer.json')))
    examples=[];hidden=[];embeddings=[]
    with torch.no_grad():
        for eid,text in [('france','The capital of France is'),('code','def add(a, b): return a + b')]:
            ids=tokenizer.encode(text).ids;tokens=[tokenizer.decode([i],skip_special_tokens=False) for i in ids]
            x=torch.stack([cp.tensor('model.embed_tokens.weight',row=i) for i in ids])[None]
            hidden.append(x);embeddings.append(x)
            examples.append({'id':eid,'label':text,'input':{'text':text},'tokens':tokens,'tokenIds':ids,'layers':[],'verification':[]})
        for layer_idx in range(6):
            prefix=f'model.layers.{layer_idx}.';is_hybrid=layer_idx==5
            with torch.device('meta'):
                mamba=zamba.Zamba2MambaDecoderLayer(config,layer_idx=layer_idx)
                module=zamba.Zamba2HybridLayer(zamba.Zamba2AttentionDecoderLayer(config,block_id=0),nn.Linear(2048,2048,bias=False),mamba) if is_hybrid else mamba
            module.load_state_dict({k.removeprefix(prefix):cp.tensor(k) for k in names if k.startswith(prefix)},strict=True,assign=True)
            module.eval();mixer=mamba.mamba
            cap={};handles=[]
            def save(name,where='out'):
                def hook(mod,args,out):
                    cap[name]=args[0].detach() if where=='in' else (out if isinstance(out,tuple) else out.detach())
                return hook
            def hook(m,name,where='out'):handles.append(m.register_forward_hook(save(name,where)))
            hook(mixer.in_proj,'mamba_input','in');hook(mixer.in_proj,'mamba_projection');hook(mixer.conv1d,'conv_raw');hook(mixer.norm,'scan_output','in');hook(mixer.norm,'gated');hook(mixer,'mamba_output')
            if is_hybrid:
                shared=module.shared_transformer
                hook(shared.input_layernorm,'attention_input');hook(shared.self_attn,'attention_result');hook(shared.feed_forward,'ffn_input','in');hook(shared.feed_forward,'ffn_output');hook(module.linear,'transformer_output')
            for ei,example in enumerate(examples):
                cap.clear();tokens=example['tokens'];n=len(tokens);x=hidden[ei]
                mask=torch.triu(torch.full((n,n),torch.finfo(torch.float32).min),diagonal=1)[None,None]
                pos=torch.arange(n)[None];position_embeddings=zamba.Zamba2RotaryEmbedding(config)(x,pos)
                out=module(x,original_hidden_states=embeddings[ei],layer_idx=layer_idx,causal_mask=mask,output_attentions=True,position_embeddings=position_embeddings)[0]
                operations=[];checks=[]
                if layer_idx==0:
                    operations.append(operation('embedding','Token embedding','embedding','Real token embedding rows from the pretrained Zamba2 checkpoint enter the Mamba backbone. This trace includes all five initial Mamba blocks and the first hybrid block.','h₀ = E[token]',[sample('Token embeddings',x,tokens)],[{'label':'Hidden width','value':2048}]))
                if is_hybrid:
                    attn=shared.self_attn;inp=cap['attention_input'];shape=(1,n,32,128)
                    q=(attn.q_proj(inp)+attn.linear_q_adapter_list[0](inp)).view(shape).transpose(1,2)
                    k=(attn.k_proj(inp)+attn.linear_k_adapter_list[0](inp)).view(shape).transpose(1,2)
                    v=(attn.v_proj(inp)+attn.linear_v_adapter_list[0](inp)).view(shape).transpose(1,2)
                    q,k=zamba.apply_rotary_pos_emb(q,k,*position_embeddings)
                    recorded_attention,weights=cap['attention_result']
                    rebuilt_weights=torch.softmax((q@k.transpose(-1,-2))*attn.scaling+mask,dim=-1)
                    context=weights@v
                    rebuilt=attn.o_proj(context.transpose(1,2).reshape(1,n,4096))
                    attn_err=float((rebuilt-recorded_attention).abs().max());weight_err=float((rebuilt_weights-weights).abs().max())
                    assert attn_err<1e-5 and weight_err<1e-6
                    checks.extend([{'name':'Transformer attention reconstruction','maxAbsoluteError':attn_err,'tolerance':1e-5},{'name':'Attention matrix reconstruction','maxAbsoluteError':weight_err,'tolerance':1e-6},{'name':'Causal attention mask','maxAbsoluteError':float(weights.triu(1).abs().max()),'tolerance':0}])
                    operations.append(operation('attention','Shared Transformer attention','attention','Block 6 concatenates the current Mamba representation with the original embeddings, applies RMSNorm, and reads causal context with a shared Transformer. Layer-specific low-rank adapters are included.','r = RMSNorm([h₅; E]); A = softmax(RoPE(Q) RoPE(K)ᵀ / √64 + mask); o = Wᴼ(A V)',[sample('Concatenated normalized input',inp,tokens),sample('Head 1 V',v[0,0],tokens)]+[tensor(f'Head {h+1} attention',weights[0,h],tokens,columnLabels=tokens,role='attention',head=h) for h in range(32)]+[sample('Head 1 attention × V',context[0,0],tokens),sample('Transformer attention output',recorded_attention,tokens)],[{'label':'Transformer heads','value':32},{'label':'Q/K/V head width','value':128},{'label':'Attention input width','value':4096}]))
                    f=shared.feed_forward;gu=f.gate_up_proj(cap['ffn_input'])+f.gate_up_proj_adapter_list[0](cap['ffn_input']);g,u=gu.chunk(2,dim=-1);product=f.act_fn(g)*u
                    ffn_err=float((f.down_proj(product)-cap['ffn_output']).abs().max());assert ffn_err<1e-5
                    checks.append({'name':'Gated GELU feed-forward reconstruction','maxAbsoluteError':ffn_err,'tolerance':1e-5})
                    operations.append(operation('feed-forward','Gated GELU feed-forward','feed-forward','The shared Transformer uses a gated GELU feed-forward network plus its layer adapter. A 2048D linear projection feeds this result into the subsequent Mamba block; the residual still uses h₅.','f = W_down(GELU(g) ⊙ u); t = W_linear f; Mamba input = RMSNorm(h₅ + t)',[sample('Gate g',g,tokens),sample('Up u',u,tokens),sample('GELU(g) × u',product,tokens),sample('Transformer contribution t',cap['transformer_output'],tokens)],[{'label':'Feed-forward width','value':8192}]))
                mops,mchecks=capture_mamba(mixer,cap,tokens);operations.extend(mops);checks.extend(mchecks)
                residual_error=float((x+cap['mamba_output']-out).abs().max());assert residual_error==0
                checks.append({'name':'Mamba residual addition','maxAbsoluteError':residual_error,'tolerance':0})
                operations.append(operation('output','Residual output','output','The Mamba output is projected to 2048D and added to the block’s original residual. '+('The trace stops here, before layers 7–38 and the LM head; no next-token prediction is claimed.' if is_hybrid else 'This exact output becomes the next recorded block’s input.'),'h_next = h + W_out RMSNorm(y ⊙ SiLU(z))',[sample('Mamba projected output',cap['mamba_output'],tokens),sample('Residual block output',out,tokens)],[{'label':'Output width','value':2048}]))
                example['layers'].append({'index':layer_idx,'label':f'Block {layer_idx+1} · '+('Transformer + Mamba2' if is_hybrid else 'Mamba2'),'operations':operations})
                example['verification'].extend([{**check,'layer':layer_idx} for check in checks]);hidden[ei]=out
                print(f'Captured {example["id"]} layer {layer_idx+1}; checks max {max(c["maxAbsoluteError"] for c in checks):.3g}',flush=True)
            for handle in handles:handle.remove()
            del module,mamba,mixer;gc.collect()
        result={'schemaVersion':1,'architecture':'hybrid','title':'Transformer–Mamba hybrid',
          'provenance':cp.provenance('Exact pretrained Zamba2-1.2B prefix: embedding, five Mamba2 blocks, then the first shared Transformer + Mamba2 hybrid block. Stops before the remaining 32 blocks and LM head; no generated prediction.',__file__,digest(Path(inspect.getfile(zamba)).read_bytes())),
          'config':{'hiddenSize':2048,'layers':38,'capturedLayers':6,'layerTypes':raw['layers_block_type'],'attentionHeads':32,'headDim':128,'mambaHeads':64,'mambaHeadDim':64,'stateSize':128,'convolutionWidth':4,'executionChunkSize':16,'checkpointChunkSize':256},'examples':examples}
        write_trace(args.output,result)

if __name__=='__main__':main()
