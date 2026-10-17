#!/usr/bin/env python3
"""Losslessly record all SmolVLM vision and language blocks with pretrained weights.

The supported variable-resolution vision encoder receives one 256px image,
without crops. Its 256 patches become 16 image tokens. This is a complete
12-block vision + connector + 30-block language forward pass, not a prefix.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from huggingface_hub import snapshot_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'HuggingFaceTB/SmolVLM-256M-Instruct'
REVISION = '7e3e67edbbed1bf9888184d9df282b700a323964'
WEIGHTS_SHA = '74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def save_tensors(path, tensors):
    path.parent.mkdir(parents=True,exist_ok=True)
    offset,entries=0,{}
    with path.open('wb') as stream:
        for name,value in tensors.items():
            a=value.detach().cpu().contiguous().numpy().astype('<f4',copy=False)
            assert np.isfinite(a).all(),name
            entries[name]={'offset':offset,'length':a.size,'shape':list(a.shape)}
            stream.write(a.tobytes()); offset+=a.size
    return {'file':path.name,'sha256':sha(path),'bytes':path.stat().st_size,'tensors':entries}

def metadata(config, phase, parameters):
    vision=phase=='vision'
    heads=config.num_attention_heads
    return {'architecture':'vision-language','maskMode':'bidirectional' if vision else 'causal',
        'normalization':'layernorm' if vision else 'rmsnorm','outputKind':'language',
        'id':MODEL,'name':'SmolVLM · '+phase,'revision':REVISION,'url':f'https://huggingface.co/{MODEL}/tree/{REVISION}',
        'license':'apache-2.0','numLayers':config.num_hidden_layers,'numExperts':0,'topK':0,
        'hiddenSize':config.hidden_size,'intermediateSize':config.intermediate_size,
        'totalParameters':parameters,'activeParametersApprox':parameters,'residualMultiplier':1,
        'attentionHeads':heads,'keyValueHeads':heads if vision else config.num_key_value_heads,
        'attentionHeadSize':64,'valueHeadSize':64,'attentionMultiplier':1/8,
        'embeddingMultiplier':1,'logitsScaling':1,'ropeTheta':0 if vision else config.rope_theta,
        'vocabSize':49280,'attentionMaskValue':0 if vision else torch.finfo(torch.float32).min,
        'rmsNormEps':config.layer_norm_eps if vision else config.rms_norm_eps,
        'activation':config.hidden_act,'denseLayers':config.num_hidden_layers,'visionLayers':12,'decoderLayers':30}

def capture(cache_dir):
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    snapshot=Path(snapshot_download(MODEL,revision=REVISION,cache_dir=cache_dir,local_files_only=True))
    assert sha(snapshot/'model.safetensors')==WEIGHTS_SHA
    image_path=ROOT/'static/architectures/vision-language-shapes.png'
    image=Image.open(image_path).convert('RGB')
    processor=AutoProcessor.from_pretrained(snapshot,local_files_only=True)
    processor.image_processor.do_image_splitting=False
    model=Idefics3ForConditionalGeneration.from_pretrained(snapshot,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='eager').eval()
    prompt='Name the colors shown in the image. Answer briefly.'
    messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
    formatted=processor.apply_chat_template(messages,add_generation_prompt=True)
    inputs=processor(text=formatted,images=[image],return_tensors='pt',image_seq_len=16,
        size={'longest_edge':256},max_image_size={'longest_edge':256})
    assert list(inputs['pixel_values'].shape[-2:])==[256,256]
    vision=model.model.vision_model
    text=model.model.text_model
    baseline=model(**inputs,output_attentions=True,output_hidden_states=True,use_cache=False)
    records,handles={},[]
    def hook(name):
        def record(module,args,output):
            result=output[0] if isinstance(output,tuple) else output
            records[name]=(args[0].detach().clone() if args else None,result.detach().clone())
            if isinstance(output,tuple) and len(output)>1 and torch.is_tensor(output[1]):
                records[name+':second']=output[1].detach().clone()
        return record
    for name,module in [('patches',vision.embeddings.patch_embedding),('positions',vision.embeddings.position_embedding),
        ('vision-embedding',vision.embeddings),('vision-final',vision.post_layernorm),
        ('projector',model.model.connector.modality_projection.proj),('language-final',text.norm),
        ('rotary',text.rotary_emb),('lm-head',model.lm_head)]:
        handles.append(module.register_forward_hook(hook(name)))
    for phase,blocks in [('vision',vision.encoder.layers),('language',text.layers)]:
        for i,layer in enumerate(blocks):
            if phase=='vision':
                parts={'norm1':layer.layer_norm1,'norm2':layer.layer_norm2,'query':layer.self_attn.q_proj,
                    'key':layer.self_attn.k_proj,'value':layer.self_attn.v_proj,'attentionOutput':layer.self_attn.out_proj,
                    'up':layer.mlp.fc1,'down':layer.mlp.fc2,'block':layer}
            else:
                parts={'norm1':layer.input_layernorm,'norm2':layer.post_attention_layernorm,'query':layer.self_attn.q_proj,
                    'key':layer.self_attn.k_proj,'value':layer.self_attn.v_proj,'attentionOutput':layer.self_attn.o_proj,
                    'up':layer.mlp.up_proj,'gate':layer.mlp.gate_proj,'down':layer.mlp.down_proj,'block':layer}
            for name,module in parts.items(): handles.append(module.register_forward_hook(hook(f'{phase}:{i}:{name}')))
    vision.config.output_attentions=True
    traced=model(**inputs,output_attentions=True,output_hidden_states=True,use_cache=False)
    for handle in handles: handle.remove()
    assert torch.equal(baseline.logits,traced.logits),'Capture changed final logits'
    tokens=[];image_indices=[]
    for pos,id in enumerate(inputs['input_ids'][0].tolist()):
        if id==model.config.image_token_id:
            tokens.append(f'image {len(image_indices)}');image_indices.append(pos)
        else: tokens.append(processor.tokenizer.decode([id],skip_special_tokens=False))
    assert len(image_indices)==16
    logits=traced.logits[0,-1]
    checks={'hookedVsUnhookedLogitMaxError':0.,'attentionQKMaxError':0.,'attentionContextMaxError':0.,
        'attentionRowSumMaxError':0.,'attentionResidualMaxError':0.,'feedForwardResidualMaxError':0.,
        'feedForwardActivationMaxError':0.,'projectorMaxError':0.,'imageMergeMaxError':0.}
    root=ROOT/'static/traces/vlm';root.mkdir(parents=True,exist_ok=True)
    total_parameters=sum(p.numel() for p in model.parameters())
    streams={}
    cos=records['rotary'][1];sin=records['rotary:second']
    for phase,blocks in [('vision',vision.encoder.layers),('language',text.layers)]:
        is_vision=phase=='vision';n=256 if is_vision else len(tokens)
        config=vision.config if is_vision else text.config
        heads=config.num_attention_heads;kvheads=heads if is_vision else config.num_key_value_heads
        entries=[];outdir=root/phase
        for i,layer in enumerate(blocks):
            r=lambda name,side=1:records[f'{phase}:{i}:{name}'][side][0]
            q=r('query').reshape(n,heads,64).transpose(0,1).contiguous()
            k=r('key').reshape(n,kvheads,64).transpose(0,1).contiguous()
            v=r('value').reshape(n,kvheads,64).transpose(0,1).contiguous()
            q_before,k_before=q,k
            if not is_vision:
                qr,kr=apply_rotary_pos_emb(q[None],k[None],cos,sin)
                q,k=qr[0],kr[0]
            expanded_key=k.repeat_interleave(heads//kvheads,dim=0)
            expanded_value=v.repeat_interleave(heads//kvheads,dim=0)
            scores=q@expanded_key.transpose(-1,-2)
            scaled=scores/8
            if not is_vision: scaled=scaled.masked_fill(torch.ones(n,n,dtype=torch.bool).triu(1),torch.finfo(torch.float32).min)
            probs=records[f'{phase}:{i}:block:second'][0] if is_vision else traced.attentions[i][0]
            context=probs@expanded_value
            tests={'attentionQKMaxError':(scaled.softmax(-1)-probs).abs().max(),
                'attentionContextMaxError':(context.transpose(0,1).reshape(n,heads*64)-r('attentionOutput',0)).abs().max(),
                'attentionRowSumMaxError':(probs.sum(-1)-1).abs().max(),
                'attentionResidualMaxError':(r('block',0)+r('attentionOutput')-r('norm2',0)).abs().max(),
                'feedForwardResidualMaxError':(r('norm2',0)+r('down')-r('block')).abs().max()}
            tensors={'query':q,'key':k,'value':v,'scores':scores,'attention':probs,'context':context,
                'input':r('block',0),'norm1':r('norm1'),'attentionOutput':r('attentionOutput'),
                'residualAfterAttention':r('norm2',0),'norm2':r('norm2'),'up':r('up'),
                'down':r('down'),'output':r('block')}
            if is_vision:
                activation=layer.mlp.activation_fn(r('up'))
                tensors['gelu']=r('down',0)
                tests['feedForwardActivationMaxError']=(activation-r('down',0)).abs().max()
                linears={'query':layer.self_attn.q_proj,'key':layer.self_attn.k_proj,'value':layer.self_attn.v_proj,
                    'attentionOutput':layer.self_attn.out_proj,'up':layer.mlp.fc1,'down':layer.mlp.fc2}
                norm_modules={'norm1':layer.layer_norm1,'norm2':layer.layer_norm2}
            else:
                tensors.update({'queryBefore':q_before,'keyBefore':k_before,'gate':r('gate'),
                    'siluGate':layer.mlp.act_fn(r('gate')),'product':r('down',0)})
                tests['feedForwardActivationMaxError']=(layer.mlp.act_fn(r('gate'))*r('up')-r('down',0)).abs().max()
                linears={'query':layer.self_attn.q_proj,'key':layer.self_attn.k_proj,'value':layer.self_attn.v_proj,
                    'attentionOutput':layer.self_attn.o_proj,'up':layer.mlp.up_proj,'gate':layer.mlp.gate_proj,'down':layer.mlp.down_proj}
                norm_modules={'norm1':layer.input_layernorm,'norm2':layer.post_attention_layernorm}
            for name,linear in linears.items():
                tensors[name+'Weights']=linear.weight[:4]
                if linear.bias is not None:tensors[name+'Bias']=linear.bias[:4]
            for name,norm in norm_modules.items():
                tensors[name+'Weight']=norm.weight
                if getattr(norm,'bias',None) is not None:tensors[name+'Bias']=norm.bias
            for name,error in tests.items(): checks[name]=max(checks[name],float(error))
            entries.append({'index':i,**save_tensors(outdir/f'block-{i:02d}.f32',tensors)})
            print(f'Captured {phase} {i+1}/{len(blocks)}',flush=True)
        if is_vision:
            common_values={'patchTokens':records['patches'][1][0].flatten(1).transpose(0,1),
                'positionEmbedding':records['positions'][1][0],
                'embedding':records['vision-embedding'][1][0],
                'finalHidden':records['vision-final'][1][0],
                'pixelValues':inputs['pixel_values'][0,0]}
        else:
            common_values={'embedding':traced.hidden_states[0][0],'finalHidden':records['language-final'][1][0],
                'logits':logits,'cos':cos[0,:,:32],'sin':sin[0,:,:32]}
            top_ids=logits.topk(4).indices
            common_values['classifierWeights']=model.lm_head.weight[top_ids]
            common_values['classifierTokenIds']=top_ids.float()
        streams[phase]={'model':metadata(config,phase,total_parameters),'sequenceLength':n,
            'tokens':[f'patch {i//16},{i%16}' for i in range(n)] if is_vision else tokens,
            'common':save_tensors(outdir/'model.f32',common_values),'layers':entries}
    packed,projected=records['projector']
    rebuilt=torch.nn.functional.linear(packed,model.model.connector.modality_projection.proj.weight)
    checks['projectorMaxError']=float((rebuilt-projected).abs().max())
    checks['imageMergeMaxError']=float((traced.hidden_states[0][0,image_indices]-projected[0]).abs().max())
    assert all(v<3e-5 for v in checks.values()),checks
    connector=save_tensors(root/'connector.f32',{'packed':packed[0],'projected':projected[0],
        'weights':model.model.connector.modality_projection.proj.weight[:4]})
    generated=model.generate(**inputs,max_new_tokens=40,do_sample=False)
    completion=generated[0,inputs['input_ids'].shape[1]:]
    vocabulary=[processor.tokenizer.decode([id],skip_special_tokens=False) for id in range(logits.numel())]
    document={'schemaVersion':1,'kind':'vlm-full','model':streams['language']['model'],'streams':streams,
        'prompt':prompt,'formattedPrompt':formatted,'tokenIds':inputs['input_ids'][0].tolist(),
        'vocabulary':vocabulary,'imageTokenPositions':image_indices,'connector':connector,
        'image':{'src':'architectures/vision-language-shapes.png','alt':'A red circle, a blue square and a green triangle on white.',
            'sha256':sha(image_path),'inputSize':256,'sourceSize':512,'patchSize':16,'gridSize':16,'pixelShuffleFactor':4},
        'completion':processor.tokenizer.decode(completion,skip_special_tokens=True),'completionIds':completion.tolist(),
        'validation':checks,'capture':{'timestamp':datetime.now(timezone.utc).isoformat(),'device':'cpu','dtype':'float32',
            'python':platform.python_version(),'packages':{name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','Pillow')},
            'script':'scripts/capture-vlm-full.py','scriptSha256':sha(__file__),'weightsSha256':WEIGHTS_SHA,
            'notes':'Complete pretrained vision encoder, pixel-shuffle connector, and language decoder. Supported variable-resolution input is explicitly resized to 256×256, giving 256 patches and 16 image tokens. All layers/heads/positions are saved losslessly; no weight or layer pruning. The recorded completion is the actual model response.'}}
    path=root/'manifest.json';path.write_text(json.dumps(document,separators=(',',':'),ensure_ascii=False)+'\n')
    print(json.dumps({'manifest':str(path),'checks':checks,'completion':document['completion'],
        'bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache-dir',default='/private/tmp/moe-hf-cache')
    capture(parser.parse_args().cache_dir)
