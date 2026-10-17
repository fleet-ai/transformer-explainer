import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { readRecordedJson } from '../src/lib/recorded-json.js';

const root = new URL('../static/traces/', import.meta.url);
const document = JSON.parse(readFileSync(new URL('deepseek-mla.json', root), 'utf8'));
const example = document.examples[0];
const operations = JSON.parse(readFileSync(new URL(`expert-operations-${example.id}.json`, root), 'utf8'));
const attentionUrl = new URL(`attention-operations-${example.id}.json`, root);
const attention = await readRecordedJson({
  ok: true, url: attentionUrl.href,
  json: async () => JSON.parse(readFileSync(attentionUrl, 'utf8')),
}, async url => ({ ok: true, arrayBuffer: async () => Uint8Array.from(readFileSync(new URL(url))).buffer }));
const near = (actual, expected, tolerance = 1e-5) => assert.ok(
  Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
  `${actual} differs from ${expected}`
);
const sum = values => values.reduce((a, b) => a + b, 0);

test('DeepSeek recording executes the complete pretrained LLM and every vocabulary logit', () => {
  assert.equal(document.provenance.partial, false);
  const sha = bytes => createHash('sha256').update(bytes).digest('hex');
  assert.equal(document.capture.scriptSha256, sha(readFileSync(new URL('../scripts/full_mla_capture.py', import.meta.url))));
  assert.equal(document.provenance.captureScriptSha256, document.capture.scriptSha256);
  assert.equal(attention.capture.mainBundleSha256, sha(readFileSync(new URL('deepseek-mla.json', root))));
  assert.equal(attention.capture.mainCaptureScriptSha256, document.capture.scriptSha256);
  assert.equal(document.model.architecture, 'mla');
  assert.equal(document.model.numLayers, 27);
  assert.equal(example.tokens[0].id, 100000);
  assert.equal(example.tokens.slice(1).map(token => token.text).join(''), example.prompt);
  assert.equal(example.layers.length, 27);
  assert.equal(example.layers[0].feedForwardType, 'dense');
  assert.ok(example.layers.slice(1).every(layer => layer.feedForwardType === 'moe'));
  assert.equal(example.validation.hookedVsUnhookedMaxLogitError, 0);
  assert.equal(example.nextTokenLogits.length, 102400);
  assert.equal(document.vocabulary.length, 102400);
  const maximum = Math.max(...example.nextTokenLogits);
  const denominator = sum(example.nextTokenLogits.map(logit => Math.exp(logit - maximum)));
  for (const candidate of example.nextTokenCandidates) {
    assert.equal(candidate.text, document.vocabulary[candidate.id]);
    near(candidate.probability, Math.exp(example.nextTokenLogits[candidate.id] - maximum) / denominator, 1e-12);
  }
});

test('Every DeepSeek layer contains all 16 causal attention heads with actual scaled scores', () => {
  const n = example.tokens.length;
  for (let layer = 0; layer < 27; layer++) {
    assert.equal(example.attention[layer].length, 16);
    for (let head = 0; head < 16; head++) {
      assert.equal(example.attention[layer][head].length, n);
      for (let query = 0; query < n; query++) {
        const row = example.attention[layer][head][query];
        const scores = example.attentionMaskedScores[layer][head][query];
        assert.ok(row.every(Number.isFinite));
        near(sum(row), 1, 1e-6);
        const max = Math.max(...scores);
        const exp = scores.map(value => Math.exp(value - max));
        const denominator = sum(exp);
        row.forEach((probability, key) => {
          near(probability, exp[key] / denominator, 2e-7);
          if (key > query) assert.equal(probability, 0);
          else assert.equal(scores[key], example.attentionScores[layer][head][query][key]);
        });
      }
    }
  }
});

test('DeepSeek preserves unnormalized top-six routing and adds its always-on shared experts', () => {
  for (const layer of example.layers.slice(1)) {
    const counts = Array(64).fill(0);
    for (const token of layer.tokens) {
      assert.equal(token.selectedExperts.length, 6);
      const expected = token.routerProbabilities.map((p, id) => ({p, id})).sort((a, b) => b.p - a.p).slice(0, 6).map(x => x.id);
      assert.deepEqual([...token.selectedExperts].sort((a,b) => a-b), expected.sort((a,b) => a-b));
      assert.ok(sum(token.routingWeights) < 1);
      near(sum(token.routerProbabilities), 1, 1e-6);
      token.selectedExperts.forEach((expert, rank) => {
        near(token.routingWeights[rank], token.routerProbabilities[expert], 1e-12);
        counts[expert]++;
      });
      for (let coordinate = 0; coordinate < 8; coordinate++) {
        const routed = sum(token.expertOutputPreview.map((values, rank) => values[coordinate] * token.routingWeights[rank]));
        near(routed, token.routedCombinedOutputPreview[coordinate]);
        near(routed + token.sharedExpertOutputPreview[coordinate], token.combinedOutputPreview[coordinate]);
      }
    }
    assert.deepEqual(layer.expertTokenCounts, counts);
  }
});

test('MLA uses actual content, latent, value, and reordered rotary coordinates in every layer', () => {
  assert.equal(attention.layers.length, 27);
  assert.equal(attention.model.keyValueHeads, 1);
  for (const layer of attention.layers) {
    assert.equal(layer.queryBefore.length, 16);
    assert.equal(layer.keyBefore.length, 1);
    assert.equal(layer.value.length, 16);
    layer.kvLatent.forEach(row => assert.equal(row.length, 512));
    for (let head = 0; head < 16; head++) {
      for (let token = 0; token < example.tokens.length; token++) {
        assert.equal(layer.queryContent[head][token].length, 128);
        assert.equal(layer.keyContent[head][token].length, 128);
        assert.equal(layer.value[head][token].length, 128);
        for (let coordinate = 0; coordinate < 32; coordinate++) {
          const c = attention.rotary.cos[token][coordinate], s = attention.rotary.sin[token][coordinate];
          const before = layer.queryBefore[head][token], after = layer.queryAfter[head][token];
          assert.equal(before[coordinate], layer.rawQueryRotary[head][token][coordinate * 2]);
          assert.equal(before[coordinate + 32], layer.rawQueryRotary[head][token][coordinate * 2 + 1]);
          near(after[coordinate], before[coordinate] * c - before[coordinate + 32] * s);
          near(after[coordinate + 32], before[coordinate + 32] * c + before[coordinate] * s);
          if (head === 0) {
            const kb = layer.keyBefore[0][token], ka = layer.keyAfter[0][token];
            near(ka[coordinate], kb[coordinate] * c - kb[coordinate + 32] * s);
            near(ka[coordinate + 32], kb[coordinate + 32] * c + kb[coordinate] * s);
          }
        }
      }
    }
  }
});

test('Recorded dense, routed, and shared expert arithmetic includes all omitted inputs', () => {
  for (const layer of operations.layers) {
    for (const token of layer.tokens) {
      const dense = layer.index === 0;
      assert.equal(token.experts.length, dense ? 1 : 7);
      assert.ok(token.experts.some(expert => expert.id === (dense ? -1 : 64)));
      for (const expert of token.experts) {
        assert.equal(expert.intermediateSize, expert.id === -1 ? 10944 : expert.id === 64 ? 2816 : 1408);
        for (const name of ['gate', 'up', 'down']) {
          const projection = expert[name];
          const input = name === 'down' ? expert.product : token.inputPreview;
          for (let row = 0; row < 4; row++) {
            const value = sum(projection.weights[row].map((weight, column) => weight * input[column])) + projection.omittedContribution[row];
            near(projection.output[row], value, 4e-4);
          }
        }
        for (let i = 0; i < 4; i++) {
          const g = expert.gate.output[i];
          near(expert.siluGate[i], g / (1 + Math.exp(-g)));
          near(expert.product[i], expert.siluGate[i] * expert.up.output[i]);
        }
      }
    }
  }
});

test('Final LM-head rows reconstruct actual full-model logits with omitted-input contributions', () => {
  const head = attention.lmHead;
  assert.equal(head.tokenIndex, example.tokens.length - 1);
  assert.equal(head.inputSize, 2048);
  assert.equal(head.tokenIds.length, 4);
  head.tokenIds.forEach((tokenId, row) => {
    const reconstructed = sum(head.weights[row].map((weight, column) => weight * head.input[column])) + head.omittedContribution[row];
    near(head.output[row], reconstructed, 1e-5);
    assert.equal(head.output[row], example.nextTokenLogits[tokenId]);
  });
});

test('Every MLA head reconstructs content-plus-position scores and attention-weighted values', () => {
  for (const layer of attention.layers) {
    const block = example.layers[layer.index];
    if (layer.index > 0) {
      assert.deepEqual(block.residualInputPreview, example.layers[layer.index - 1].blockOutputPreview);
    }
    for (let head = 0; head < 16; head++) {
      for (let query = 0; query < example.tokens.length; query++) {
        assert.deepEqual(block.queryPreview[head][query], layer.queryContent[head][query].slice(0, 8));
        assert.deepEqual(block.keyPreview[head][query], layer.keyContent[head][query].slice(0, 8));
        assert.deepEqual(block.valuePreview[head][query], layer.value[head][query].slice(0, 8));
        for (let key = 0; key < example.tokens.length; key++) {
          const content = sum(layer.queryContent[head][query].map((q, i) => q * layer.keyContent[head][key][i]));
          const rotary = sum(layer.queryAfter[head][query].map((q, i) => q * layer.keyAfter[0][key][i]));
          near(example.attentionScores[layer.index][head][query][key], (content + rotary) * document.model.attentionMultiplier, 2e-5);
        }
        for (let coordinate = 0; coordinate < 128; coordinate++) {
          const weighted = sum(example.attention[layer.index][head][query].map((p, key) => p * layer.value[head][key][coordinate]));
          near(layer.attentionContext[head][query][coordinate], weighted, 2e-5);
        }
      }
    }
    assert.deepEqual(block.attentionOutputPreview, layer.attentionOutput.map(row => row.slice(0, 8)));
  }
});
