#!/usr/bin/env python3
"""Record the complete pretrained DeiT-tiny forward pass, all blocks and heads.

Float32 tensors are stored losslessly in little-endian binary files. The UI may
show a position window, but attention softmax and A@V always use all 197 tokens.
No model weights are generated or initialized by this script.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'facebook/deit-tiny-patch16-224'
REVISION = 'b3428f18dcc7b543470d07f14b4a4157815d1880'
WEIGHTS_SHA = 'e1a51b0c81ff812e079d2189352747947879fee56429e4480064a6695cb34b2c'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def save_tensors(path, tensors):
    offset, entries = 0, {}
    with path.open('wb') as f:
        for name, value in tensors.items():
            a=value.detach().cpu().contiguous().numpy().astype('<f4',copy=False)
            assert np.isfinite(a).all(), name
            entries[name]={'offset':offset,'length':a.size,'shape':list(a.shape)}
            f.write(a.tobytes()); offset += a.size
    return {'file':path.name,'sha256':sha(path),'bytes':path.stat().st_size,'tensors':entries}

def capture(cache_dir):
    torch.set_num_threads(4)
    snapshot=Path(snapshot_download(MODEL,revision=REVISION,cache_dir=cache_dir,local_files_only=True))
    assert sha(snapshot/'pytorch_model.bin')==WEIGHTS_SHA
    from PIL import Image
    image_path=ROOT/'static/architectures/vit-shapes.png'
    image=Image.open(image_path).convert('RGB')
    processor=AutoImageProcessor.from_pretrained(snapshot,local_files_only=True,use_fast=False)
    model=AutoModelForImageClassification.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    inputs=processor(images=image,return_tensors='pt')
    records, handles={},[]
    def hook(name):
        def fn(_,args,out):
            records[name]=(args[0].detach().clone(), (out[0] if isinstance(out,tuple) else out).detach().clone())
        return fn
    with torch.inference_mode():
        baseline=model(**inputs,output_attentions=True,output_hidden_states=True)
        handles.append(model.vit.embeddings.patch_embeddings.register_forward_hook(hook('patches')))
        handles.append(model.vit.layernorm.register_forward_hook(hook('finalnorm')))
        for i,layer in enumerate(model.vit.encoder.layer):
            for name,module in {'norm1':layer.layernorm_before,'q':layer.attention.attention.query,'k':layer.attention.attention.key,'v':layer.attention.attention.value,'out':layer.attention.output.dense,'norm2':layer.layernorm_after,'up':layer.intermediate.dense,'gelu':layer.intermediate,'down':layer.output.dense,'block':layer}.items():
                handles.append(module.register_forward_hook(hook(f'{i}:{name}')))
        traced=model(**inputs,output_attentions=True,output_hidden_states=True)
        for handle in handles: handle.remove()
        outdir=ROOT/'static/traces/vit';outdir.mkdir(parents=True,exist_ok=True)
        checks={'hookedVsUnhookedLogitMaxError':(baseline.logits-traced.logits).abs().max().item(), 'attentionQKMaxError':0.,'attentionContextMaxError':0.,'attentionRowSumMaxError':0.,'attentionResidualMaxError':0.,'feedForwardResidualMaxError':0.,'classifierProjectionMaxError':0.}
        layers=[]
        for i,layer in enumerate(model.vit.encoder.layer):
            r=lambda name,side=1:records[f'{i}:{name}'][side][0]
            q,k,v=[r(name).reshape(197,3,64).transpose(0,1).contiguous() for name in ('q','k','v')]
            scores=q@k.transpose(-1,-2)
            probs=traced.attentions[i][0]
            context=(probs@v)
            checks['attentionQKMaxError']=max(checks['attentionQKMaxError'],(torch.softmax(scores/8.,-1)-probs).abs().max().item())
            checks['attentionContextMaxError']=max(checks['attentionContextMaxError'],(context.transpose(0,1).reshape(197,192)-r('out',0)).abs().max().item())
            checks['attentionRowSumMaxError']=max(checks['attentionRowSumMaxError'],(probs.sum(-1)-1).abs().max().item())
            checks['attentionResidualMaxError']=max(checks['attentionResidualMaxError'],(r('block',0)+r('out')-r('norm2',0)).abs().max().item())
            checks['feedForwardResidualMaxError']=max(checks['feedForwardResidualMaxError'],(r('norm2',0)+r('down')-r('block')).abs().max().item())
            tensors={'query':q,'key':k,'value':v,'scores':scores,'attention':probs,'context':context,'input':r('block',0),'norm1':r('norm1'),'attentionOutput':r('out'),'residualAfterAttention':r('norm2',0),'norm2':r('norm2'),'up':r('up'),'gelu':r('gelu'),'down':r('down'),'output':r('block')}
            # Four output rows, all input columns: exact dot products including bias.
            for name,linear in [('query',layer.attention.attention.query),('key',layer.attention.attention.key),('value',layer.attention.attention.value),('up',layer.intermediate.dense),('down',layer.output.dense),('attentionOutput',layer.attention.output.dense)]:
                count = linear.out_features if name in ('query','key','value') else 4
                tensors[name+'Weights']=linear.weight[:count];tensors[name+'Bias']=linear.bias[:count]
            for name,norm in [('norm1',layer.layernorm_before),('norm2',layer.layernorm_after)]:
                tensors[name+'Weight']=norm.weight;tensors[name+'Bias']=norm.bias
            layers.append({'index':i,**save_tensors(outdir/f'block-{i:02d}.f32',tensors)})
        final=records['finalnorm'][1][0]
        logits=traced.logits[0]
        checks['classifierProjectionMaxError']=(model.classifier(final[0])-logits).abs().max().item()
        assert all(v<2e-5 for v in checks.values()), checks
        embeddings=torch.cat([model.vit.embeddings.cls_token.expand(1,-1,-1),records['patches'][1]],1)[0]
        common=save_tensors(outdir/'model.f32',{'patchTokens':embeddings,'positionEmbedding':model.vit.embeddings.position_embeddings[0],'embedding':traced.hidden_states[0][0],'finalHidden':final,'classifierWeights':model.classifier.weight,'classifierBias':model.classifier.bias,'logits':logits,'pixelValues':inputs['pixel_values'][0]})
        doc={'schemaVersion':1,'kind':'vit-full','model':{'architecture':'vit','maskMode':'bidirectional','normalization':'layernorm','outputKind':'classification','id':MODEL,'name':'DeiT-tiny','revision':REVISION,'url':f'https://huggingface.co/{MODEL}/tree/{REVISION}','license':'apache-2.0','numLayers':12,'numExperts':0,'topK':0,'hiddenSize':192,'intermediateSize':768,'totalParameters':sum(p.numel() for p in model.parameters()),'activeParametersApprox':sum(p.numel() for p in model.parameters()),'residualMultiplier':1,'attentionHeads':3,'keyValueHeads':3,'attentionHeadSize':64,'valueHeadSize':64,'attentionMultiplier':1/8,'embeddingMultiplier':1,'logitsScaling':1,'ropeTheta':0,'vocabSize':1000,'attentionMaskValue':0,'rmsNormEps':model.config.layer_norm_eps},'sequenceLength':197,'image':{'src':'architectures/vit-shapes.png','alt':'A red circle, blue square and green triangle on white.','patchSize':16,'gridSize':14,'sha256':sha(image_path)},'tokens':['[CLS]']+[f'patch {i//14},{i%14}' for i in range(196)],'classLabels':[model.config.id2label[i] for i in range(1000)],'common':common,'layers':layers,'validation':checks,'capture':{'timestamp':datetime.now(timezone.utc).isoformat(),'device':'cpu','dtype':'float32','python':platform.python_version(),'packages':{x:importlib.metadata.version(x) for x in ('torch','transformers','numpy','Pillow')},'script':'scripts/capture-vit-full.py','scriptSha256':sha(__file__),'weightsSha256':WEIGHTS_SHA,'notes':'Complete pretrained 12-block inference, every one of 3 heads and all 197 positions. Lossless float32 attention/Q/K/V and full hidden/FFN activations. Projection weight slices contain four output rows and every input coordinate. The input is a synthetic test image; ImageNet predictions are actual outputs, not shape labels.'}}
        path=outdir/'manifest.json';path.write_text(json.dumps(doc,separators=(',',':'))+'\n')
        print(json.dumps({'manifest':str(path),'bytes':sum(p.stat().st_size for p in outdir.iterdir()),'checks':checks}),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache-dir',default='/private/tmp/moe-hf-cache');args=parser.parse_args();capture(args.cache_dir)
