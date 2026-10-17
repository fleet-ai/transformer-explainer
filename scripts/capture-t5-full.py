#!/usr/bin/env python3
"""Capture every layer and every attention head of pretrained T5-small.

The encoder, decoder self-attention, and encoder-decoder attention execute in
one unmodified Transformers forward pass over the actual generated prefix.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODEL='google-t5/t5-small'
REVISION='df1b051c49625cf57a3d0d8d3863ed4d13564fe4'
WEIGHT_SHA='bd944e5f1b3ad9b70dd9d00010a517059e19265671076b8b0a4a58d9491842bc'

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def capture(cache_dir):
 import torch
 from huggingface_hub import snapshot_download
 from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.manual_seed(0)
 snapshot=Path(snapshot_download(MODEL,revision=REVISION,cache_dir=cache_dir,allow_patterns=['*.json','*.model','model.safetensors'],local_files_only=True))
 assert digest(snapshot/'model.safetensors')==WEIGHT_SHA
 tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
 model=AutoModelForSeq2SeqLM.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
 cfg=model.config; H=cfg.num_heads;D=cfg.d_kv;hidden=cfg.d_model
 def val(x):return x.detach().cpu().tolist()
 def preview(x):return val(x[...,:8])
 def norms(x):return val(x.float().norm(dim=-1))
 def tokens(ids):
  return [{'index':i,'id':int(n),'text':tokenizer.decode([int(n)],skip_special_tokens=False) or tokenizer.convert_ids_to_tokens(int(n))} for i,n in enumerate(ids)]
 def heads(x):return x[0].reshape(-1,H,D).transpose(0,1)
 def projection(weight,x,y):
  w=weight[:4,:4].float(); first=x[:4].float();output=y[:4].float()
  return {'weights':val(w),'output':val(output),'omittedContribution':val(output-(w@first))}
 docs=[];checks={'hookedVsUnhookedMaxLogitError':0.0,'attentionContextMaxError':0.0,'softmaxReconstructionMaxError':0.0,'decoderFutureAttentionMax':0.0,'feedForwardMaxError':0.0}
 for ei,prompt in enumerate(['translate English to German: The house is wonderful.','translate English to French: The cat is sleeping.']):
  inputs=tokenizer(prompt,return_tensors='pt');generated=model.generate(**inputs,max_new_tokens=20,do_sample=False);decoder_ids=generated[:,:-1]
  baseline=model(**inputs,decoder_input_ids=decoder_ids,use_cache=False,output_attentions=True,return_dict=True)
  observed={};hooks=[]
  def watch(module,key):
   def fn(_,args,output):observed[key]=(tuple(x.detach().clone() if isinstance(x,torch.Tensor) else x for x in args),tuple(x.detach().clone() if isinstance(x,torch.Tensor) else x for x in output) if isinstance(output,tuple) else output.detach().clone())
   hooks.append(module.register_forward_hook(fn))
  for stack_name,stack in [('encoder',model.encoder),('decoder',model.decoder)]:
   watch(stack.final_layer_norm,f'{stack_name}-finalnorm')
   for li,block in enumerate(stack.block):
    key=f'{stack_name}-{li}';watch(block,key+'-block');watch(block.layer[0],key+'-selfresidual');watch(block.layer[-1].DenseReluDense,key+'-ffn')
    for attr in ['wi','wo']:watch(getattr(block.layer[-1].DenseReluDense,attr),key+'-'+attr)
    for name,attention in [('self',block.layer[0].SelfAttention)]+([('cross',block.layer[1].EncDecAttention)] if stack_name=='decoder' else []):
     akey=f'{key}-{name}';watch(attention,akey)
     for attr in ['q','k','v','o']:watch(getattr(attention,attr),akey+'-'+attr)
    if stack_name=='decoder':watch(block.layer[1],key+'-crossresidual')
  result=model(**inputs,decoder_input_ids=decoder_ids,use_cache=False,output_attentions=True,output_hidden_states=True,return_dict=True)
  for hook in hooks:hook.remove()
  checks['hookedVsUnhookedMaxLogitError']=max(checks['hookedVsUnhookedMaxLogitError'],float((baseline.logits-result.logits).abs().max()))
  logits=result.logits[0,-1];probs=logits.double().softmax(-1);p,idx=probs.topk(16)
  candidates=[{'id':int(n),'text':tokenizer.decode([int(n)],skip_special_tokens=False),'probability':float(prob)} for prob,n in zip(p,idx)]
  def attention_record(akey):
   q,k,v=[heads(observed[akey+'-'+name][1]) for name in ['q','k','v']]
   output,past,bias,a=observed[akey][1];raw=q@k.transpose(-1,-2);masked=raw+bias[0];context=heads(observed[akey+'-o'][0][0])
   checks['attentionContextMaxError']=max(checks['attentionContextMaxError'],float((a[0]@v-context).abs().max()))
   checks['softmaxReconstructionMaxError']=max(checks['softmaxReconstructionMaxError'],float((masked.softmax(-1)-a[0]).abs().max()))
   return {'queryBefore':val(q),'queryAfter':val(q),'keyBefore':val(k),'keyAfter':val(k),'value':val(v),'attentionContext':val(context),'attentionInput':val(observed[akey][0][0][0]),'attentionOutput':val(output[0]),'scores':val(raw),'maskedScores':val(masked),'probabilities':val(a[0]),'positionBias':val(bias[0])}
  stacks={};cross=[];ffns={}
  for stack_name,stack,ids in [('encoder',model.encoder,inputs['input_ids'][0]),('decoder',model.decoder,decoder_ids[0])]:
   layer_data=[];attentions=[];scores=[];masked=[];ops=[];ffn_layers=[]
   for li,block in enumerate(stack.block):
    key=f'{stack_name}-{li}';attn=attention_record(key+'-self');ops.append({'index':li,**attn});attentions.append(attn['probabilities']);scores.append(attn['scores']);masked.append(attn['maskedScores'])
    if stack_name=='decoder':
     cross.append({'index':li,**attention_record(key+'-cross')})
     checks['decoderFutureAttentionMax']=max(checks['decoderFutureAttentionMax'],float(result.decoder_attentions[li].triu(1).abs().max()))
    x=observed[key+'-ffn'][0][0][0];wi=observed[key+'-wi'][1][0];act=observed[key+'-wo'][0][0][0];out=observed[key+'-ffn'][1][0];weight=block.layer[-1].DenseReluDense
    checks['feedForwardMaxError']=max(checks['feedForwardMaxError'],float((torch.relu(wi)-act).abs().max()),float((act@weight.wo.weight.T-out).abs().max()))
    ffn_layers.append({'index':li,'tokens':[{'inputPreview':val(x[t,:4]),'up':projection(weight.wi.weight,x[t],wi[t]),'activation':val(act[t,:4]),'down':projection(weight.wo.weight,act[t],out[t])} for t in range(len(ids))]})
    raw_input=observed[key+'-block'][0][0][0];block_out=observed[key+'-block'][1][0][0]
    layer_data.append({'index':li,'feedForwardType':'dense','tokens':[{'routerLogits':[],'routerProbabilities':[],'selectedExperts':[],'routingWeights':[],'expertOutputNorms':[],'weightedOutputNorms':[],'expertOutputPreview':[],'combinedOutputPreview':preview(out[t]),'inputNorm':float(x[t].norm()),'outputNorm':float(out[t].norm()),'reconstructionMaxAbsError':0} for t in range(len(ids))],
     'expertTokenCounts':[],'attentionInputPreview':[r[:8] for r in attn['attentionInput']],'attentionInputNorms':[sum(v*v for v in r)**.5 for r in attn['attentionInput']], 'attentionOutputPreview':[r[:8] for r in attn['attentionOutput']], 'attentionOutputNorms':[sum(v*v for v in r)**.5 for r in attn['attentionOutput']],
     'queryPreview':[[r[:8] for r in h] for h in attn['queryAfter']],'keyPreview':[[r[:8] for r in h] for h in attn['keyAfter']],'valuePreview':[[r[:8] for r in h] for h in attn['value']], 'queryNorms':norms(heads(observed[key+'-self-q'][1])),'keyNorms':norms(heads(observed[key+'-self-k'][1])),'valueNorms':norms(heads(observed[key+'-self-v'][1])),
     'residualInputPreview':preview(raw_input),'residualAfterAttentionPreview':preview(observed[key+'-selfresidual'][1][0][0]),'feedForwardInputPreview':preview(x),'blockOutputPreview':preview(block_out)})
   embedding=model.shared(ids);final=result.encoder_last_hidden_state[0] if stack_name=='encoder' else observed['decoder-finalnorm'][1][0]
   example={'id':f't5-{ei+1}-{stack_name}','title':'English → German' if ei==0 else 'English → French','category':'translation','prompt':prompt,'tokens':tokens(ids),'layers':layer_data,'attention':attentions,'attentionScores':scores,'attentionMaskedScores':masked,'embeddingPreview':preview(embedding),'embeddingNorms':norms(embedding),'finalHiddenPreview':preview(final),'finalHiddenNorms':norms(final),'nextTokenLogits':val(logits),'nextTokenCandidates':candidates,'validation':checks.copy()}
   stacks[stack_name]={'example':example,'attentionOperations':{'schemaVersion':1,'kind':'attention-operations','model':{'id':MODEL,'revision':REVISION,'architecture':'encoder-decoder','attentionHeads':H,'keyValueHeads':H,'headSize':D,'ropeTheta':0,'attentionMultiplier':1},'capture':{'notes':['T5 uses learned relative position biases; no RoPE.']},'example':{'id':example['id'],'prompt':prompt,'tokens':example['tokens']},'positions':list(range(len(ids))),'rotary':{'cos':[],'sin':[]},'layers':ops,'validation':checks.copy()}}
   ffns[stack_name]=ffn_layers
  final=observed['decoder-finalnorm'][1][0,-1];scaled=final*(hidden**-.5);topids=idx[:4].tolist();weights=model.lm_head.weight[topids,:4].float();output=logits[topids];omitted=output-(weights@scaled[:4])
  docs.append({'id':f't5-translation-{ei+1}','prompt':prompt,'translation':tokenizer.decode(generated[0],skip_special_tokens=True),'generatedTokenIds':val(generated[0]),**stacks,'crossAttention':cross,'feedForward':ffns,'lmHead':{'inputPreview':val(scaled[:4]),'unscaledInputPreview':val(final[:4]),'scale':hidden**-.5,'candidateIds':topids,'candidateLabels':[tokenizer.decode([n],skip_special_tokens=False) for n in topids],'weights':val(weights),'omittedContribution':val(omitted),'logits':val(output)}})
  print(f'Captured full T5 example {ei+1}: {docs[-1]["translation"]}',flush=True)
 assert checks['hookedVsUnhookedMaxLogitError']==0
 assert checks['decoderFutureAttentionMax']==0
 assert max(checks.values())<2e-5,checks
 metadata={'architecture':'encoder-decoder','id':MODEL,'name':'T5-small','revision':REVISION,'url':f'https://huggingface.co/{MODEL}','license':'apache-2.0','numLayers':6,'encoderLayers':6,'decoderLayers':6,'numExperts':0,'topK':0,'hiddenSize':hidden,'intermediateSize':cfg.d_ff,'denseIntermediateSize':cfg.d_ff,'totalParameters':sum(p.numel() for p in model.parameters()),'activeParametersApprox':sum(p.numel() for p in model.parameters()),'residualMultiplier':1,'attentionHeads':H,'keyValueHeads':H,'attentionHeadSize':D,'valueHeadSize':D,'attentionMultiplier':1,'embeddingMultiplier':1,'logitsScaling':hidden**-.5,'ropeTheta':0,'vocabSize':cfg.vocab_size,'attentionMaskValue':torch.finfo(torch.float32).min,'attentionBias':'relative','maskMode':'causal','rmsNormEps':cfg.layer_norm_epsilon}
 vocabulary=[tokenizer.decode([i],skip_special_tokens=False) if i<len(tokenizer) else f'<unused:{i}>' for i in range(cfg.vocab_size)]
 doc={'schemaVersion':1,'kind':'full-encoder-decoder','model':metadata,'vocabulary':vocabulary,'capture':{'timestamp':datetime.now(timezone.utc).isoformat(),'device':'cpu','dtype':'float32','python':__import__('platform').python_version(),'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','huggingface-hub']},'script':'scripts/capture-t5-full.py','scriptSha256':digest(__file__),'weightsSha256':WEIGHT_SHA,'configSha256':digest(snapshot/'config.json'),'notes':['All six encoder blocks, six decoder blocks and eight heads are captured.','Greedy generation uses the unmodified full pretrained T5 model. The recorded forward pass uses its actual generated prefix.','Every layer includes full Q/K/V and attention matrices. Only displayed residual and feed-forward arithmetic previews are sliced.','Probabilities are next-token probabilities after the generated translation prefix.']},'examples':docs,'validation':checks}
 out=ROOT/'static/traces/t5-full.json';out.write_text(json.dumps(doc,ensure_ascii=False,separators=(',',':'))+'\n')
 print(json.dumps({'output':str(out),'bytes':out.stat().st_size,'validation':checks}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-dir',default='/private/tmp/moe-hf-cache');capture(p.parse_args().cache_dir)
