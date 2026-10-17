import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
const trace=JSON.parse(readFileSync(new URL('../static/traces/t5-full.json',import.meta.url)));
const approx=(a,b,tolerance=2e-5)=>assert.ok(Math.abs(a-b)<=tolerance,`${a} != ${b}`);
const sum=a=>a.reduce((s,v)=>s+v,0);
test('T5 provenance is the complete pinned pretrained model and current capture script',()=>{
 assert.equal(trace.model.id,'google-t5/t5-small');assert.equal(trace.model.encoderLayers,6);assert.equal(trace.model.decoderLayers,6);assert.equal(trace.model.attentionHeads,8);assert.equal(trace.model.totalParameters,60506624);
 const sha=createHash('sha256').update(readFileSync(new URL('../scripts/capture-t5-full.py',import.meta.url))).digest('hex');assert.equal(trace.capture.scriptSha256,sha);
 for(const error of Object.values(trace.validation))assert.equal(error,0);
});
test('all encoder, decoder and cross-attention matrices reconstruct from all 64 head coordinates',()=>{
 for(const e of trace.examples){
  const source=e.encoder.example.tokens.length, target=e.decoder.example.tokens.length;
  assert.equal(e.encoder.example.layers.length,6);assert.equal(e.decoder.example.layers.length,6);assert.equal(e.crossAttention.length,6);
  for(const [name,layers,qs,ks] of [['encoder',e.encoder.attentionOperations.layers,source,source],['decoder',e.decoder.attentionOperations.layers,target,target],['cross',e.crossAttention,target,source]])for(const layer of layers){
   assert.equal(layer.queryAfter.length,8);assert.equal(layer.keyAfter.length,8);
   for(let h=0;h<8;h++){
    assert.equal(layer.queryAfter[h].length,qs);assert.equal(layer.keyAfter[h].length,ks);
    for(let q=0;q<qs;q++){
     const p=layer.probabilities[h][q];approx(sum(p),1,2e-7);
     const logits=layer.maskedScores[h][q], max=Math.max(...logits), denominator=sum(logits.map(v=>Math.exp(v-max)));
     for(let k=0;k<ks;k++){
      const dot=sum(layer.queryAfter[h][q].map((v,d)=>v*layer.keyAfter[h][k][d]));approx(dot,layer.scores[h][q][k],1e-4);
      approx(p[k],Math.exp(logits[k]-max)/denominator,2e-7);
      if(name==='decoder'&&k>q)assert.equal(p[k],0);
     }
     for(const d of [0,1,31,63])approx(sum(p.map((v,k)=>v*layer.value[h][k][d])),layer.attentionContext[h][q][d],2e-5);
    }
   }
  }
  assert.ok(e.encoder.attentionOperations.layers.some(l=>l.probabilities.some(h=>h[0].slice(1).some(v=>v>0))),'encoder must see future source tokens');
 }
});
test('dense FFN projections, ReLU and output vocabulary rows preserve numeric arithmetic',()=>{
 for(const e of trace.examples){
  for(const stack of ['encoder','decoder'])for(const layer of e.feedForward[stack])for(const t of layer.tokens){
   for(let i=0;i<4;i++){
    approx(sum(t.up.weights[i].map((w,d)=>w*t.inputPreview[d]))+t.up.omittedContribution[i],t.up.output[i],1e-5);
    assert.equal(t.activation[i],Math.max(0,t.up.output[i]));
    approx(sum(t.down.weights[i].map((w,d)=>w*t.activation[d]))+t.down.omittedContribution[i],t.down.output[i],1e-5);
   }
  }
  assert.equal(e.decoder.example.nextTokenLogits.length,32128);
  for(let i=0;i<4;i++){
   approx(e.lmHead.inputPreview[i],e.lmHead.unscaledInputPreview[i]*e.lmHead.scale,1e-7);
   approx(sum(e.lmHead.weights[i].map((w,d)=>w*e.lmHead.inputPreview[d]))+e.lmHead.omittedContribution[i],e.lmHead.logits[i],1e-5);
   assert.equal(e.lmHead.logits[i],e.decoder.example.nextTokenLogits[e.lmHead.candidateIds[i]]);
  }
 }
});
