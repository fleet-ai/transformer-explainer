import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const near = (actual, expected, label, absolute = 1e-5, relative = 2e-6) => {
  assert.ok(Number.isFinite(actual) && Number.isFinite(expected)
    && Math.abs(actual - expected) <= absolute + relative * Math.abs(expected), label);
};
const vector = (values, length, label) => {
  assert.ok(Array.isArray(values), `${label}: array required`);
  assert.equal(values.length, length, `${label}: coordinate count`);
  assert.ok(values.every(Number.isFinite), `${label}: non-finite coordinate`);
};

/** Check full captured heads, real half-split RoPE, GQA alignment, and attention dots. */
export function validateAttentionOperations(sidecar, main) {
  assert.equal(sidecar.schemaVersion, 1);
  assert.equal(sidecar.kind, 'attention-operations');
  for (const key of ['id', 'revision', 'attentionHeads', 'keyValueHeads', 'ropeTheta', 'attentionMultiplier']) {
    assert.equal(sidecar.model[key], main.model[key], `Model ${key}`);
  }
  assert.equal(sidecar.model.headSize, main.model.attentionHeadSize);
  assert.match(sidecar.capture.scriptSha256, /^[a-f0-9]{64}$/);
  assert.match(sidecar.capture.mainBundleSha256, /^[a-f0-9]{64}$/);
  assert.equal(sidecar.capture.mainCaptureScriptSha256, main.capture.scriptSha256);
  assert.equal(sidecar.capture.matchesMainVocabularyLogits, true);
  assert.equal(sidecar.capture.actualValueUnchangedByRotation, true);
  assert.equal(sidecar.capture.actualGroupedKeyValueRepetitionExact, true);
  assert.equal(sidecar.capture.projectionHeadReshapeExact, true);
  const example = main.examples.find(item => item.id === sidecar.example.id);
  assert.ok(example, 'Example is present in the main bundle');
  assert.equal(sidecar.example.prompt, example.prompt);
  assert.deepEqual(sidecar.example.tokens, example.tokens);
  const count = example.tokens.length;
  const { attentionHeads, keyValueHeads, headSize, attentionMultiplier, ropeTheta } = sidecar.model;
  assert.equal(attentionHeads % keyValueHeads, 0, 'Whole number of query heads per KV head');
  assert.equal(headSize % 2, 0, 'Even head size for half-split rotation');
  const groupSize = attentionHeads / keyValueHeads;
  const half = headSize / 2;
  assert.deepEqual(sidecar.positions, Array.from({ length: count }, (_, index) => index));
  assert.equal(sidecar.rotary.cos.length, count);
  assert.equal(sidecar.rotary.sin.length, count);
  for (let token = 0; token < count; token++) {
    vector(sidecar.rotary.cos[token], half, 'Cosine');
    vector(sidecar.rotary.sin[token], half, 'Sine');
    for (let pair = 0; pair < half; pair++) {
      const cosine = sidecar.rotary.cos[token][pair];
      const sine = sidecar.rotary.sin[token][pair];
      const angle = sidecar.positions[token] / Math.pow(ropeTheta, 2 * pair / headSize);
      near(cosine, Math.cos(angle), 'Actual position cosine', 2e-6);
      near(sine, Math.sin(angle), 'Actual position sine', 2e-6);
      near(cosine * cosine + sine * sine, 1, 'Unit rotation', 2e-6);
    }
  }
  assert.equal(sidecar.layers.length, main.model.numLayers);
  let queryVectors = 0;
  let keyVectors = 0;
  let valueVectors = 0;
  let rotatedPairs = 0;
  let attentionDots = 0;
  let maxRotationError = 0;
  let maxAttentionScoreError = 0;

  function validateRotation(before, after, token) {
    for (let pair = 0; pair < half; pair++) {
      const cosine = sidecar.rotary.cos[token][pair];
      const sine = sidecar.rotary.sin[token][pair];
      const a = before[pair];
      const b = before[pair + half];
      const reconstructed = [a * cosine - b * sine, b * cosine + a * sine];
      for (let offset = 0; offset < 2; offset++) {
        const actual = after[pair + offset * half];
        near(actual, reconstructed[offset], 'Half-split RoPE coordinate');
        maxRotationError = Math.max(maxRotationError, Math.abs(actual - reconstructed[offset]));
      }
      rotatedPairs++;
    }
    near(Math.hypot(...after), Math.hypot(...before), 'RoPE preserves head norm');
  }

  for (const [layerIndex, layer] of sidecar.layers.entries()) {
    assert.equal(layer.index, layerIndex, 'Layer index');
    const original = example.layers[layerIndex];
    for (const field of ['queryBefore', 'queryAfter', 'keyBefore', 'keyAfter', 'value']) {
      assert.equal(layer[field].length, field.startsWith('query') ? attentionHeads : keyValueHeads, `${field}: unique head count`);
      layer[field].forEach(head => {
        assert.equal(head.length, count, `${field}: token count`);
        head.forEach(row => vector(row, headSize, field));
      });
    }
    for (let head = 0; head < attentionHeads; head++) {
      const sharedHead = Math.floor(head / groupSize);
      for (let token = 0; token < count; token++) {
        validateRotation(layer.queryBefore[head][token], layer.queryAfter[head][token], token);
        queryVectors++;
        assert.deepEqual(layer.queryAfter[head][token].slice(0, 8), original.queryPreview[head][token], 'Actual post-RoPE query preview');
        assert.deepEqual(layer.keyAfter[sharedHead][token].slice(0, 8), original.keyPreview[head][token], 'Shared post-RoPE key preview');
        assert.deepEqual(layer.value[sharedHead][token].slice(0, 8), original.valuePreview[head][token], 'Value stays unchanged and shared');
        near(Math.hypot(...layer.queryAfter[head][token]), original.queryNorms[head][token], 'Full query norm');
        near(Math.hypot(...layer.keyAfter[sharedHead][token]), original.keyNorms[head][token], 'Full shared key norm');
        near(Math.hypot(...layer.value[sharedHead][token]), original.valueNorms[head][token], 'Full unchanged value norm');
        for (let keyToken = 0; keyToken < count; keyToken++) {
          let dot = 0;
          let absoluteTerms = 0;
          for (let coordinate = 0; coordinate < headSize; coordinate++) {
            const term = layer.queryAfter[head][token][coordinate] * layer.keyAfter[sharedHead][keyToken][coordinate];
            dot += term;
            absoluteTerms += Math.abs(term);
          }
          const expected = example.attentionScores[layerIndex][head][token][keyToken];
          const error = Math.abs(dot * attentionMultiplier - expected);
          // Float32 accumulation is bounded by the actual dot terms, including cancellation.
          assert.ok(error <= 1e-5 + 1e-6 * absoluteTerms * attentionMultiplier, 'Full QK dot matches recorded scaled score');
          maxAttentionScoreError = Math.max(maxAttentionScoreError, error);
          attentionDots++;
        }
      }
    }
    for (let head = 0; head < keyValueHeads; head++) {
      for (let token = 0; token < count; token++) {
        validateRotation(layer.keyBefore[head][token], layer.keyAfter[head][token], token);
        keyVectors++;
        valueVectors++;
      }
    }
  }
  return { queryVectors, keyVectors, valueVectors, rotatedPairs, attentionDots, maxRotationError, maxAttentionScoreError };
}

export function verifyAttentionOperationsProvenance(sidecar, mainBytes, sourceBytes) {
  assert.equal(sidecar.capture.mainBundleSha256, createHash('sha256').update(mainBytes).digest('hex'), 'Main bundle hash');
  assert.equal(sidecar.capture.scriptSha256, createHash('sha256').update(sourceBytes).digest('hex'), 'Attention capture source hash');
}

/** Validate the independent pretrained 128D Qwen example, including every rotary pair. */
export function validateRopeExample(data) {
  assert.equal(data.schemaVersion, 1);
  assert.equal(data.kind, 'rope-example');
  assert.equal(data.example.id, 'qwen3-128');
  assert.equal(data.model.id, 'Qwen/Qwen3-0.6B');
  assert.equal(data.model.revision, 'c1899de289a04d12100db370d81485cdf75e47ca');
  assert.equal(data.model.headSize, 128);
  assert.equal(data.model.attentionHeads, 16);
  assert.equal(data.model.keyValueHeads, 8);
  assert.equal(data.model.ropeTheta, 1000000);
  assert.equal(data.model.ropeScaling, null);
  assert.equal(data.model.hiddenSize, 1024);
  assert.equal(data.model.numLayers, 28);
  near(data.model.attentionMultiplier, 128 ** -0.5, 'Qwen attention scale', 1e-12, 1e-12);
  assert.deepEqual(data.scope, { layer: 0, queryHead: 0, keyValueHead: 0 });
  assert.equal(data.capture.hookedVsUnhookedMaxLogitError, 0);
  assert.equal(data.capture.actualValueUnchangedByRotation, true);
  assert.equal(data.capture.actualNormalizationOutputsMatchRotaryInputs, true);
  assert.equal(data.capture.actualGroupedKeyValueRepetitionExact, true);
  assert.equal(data.capture.queryInputNormalization, 'q_norm');
  assert.equal(data.capture.keyInputNormalization, 'k_norm');
  for (const field of ['scriptSha256', 'modelSourceSha256', 'configSha256']) {
    assert.match(data.capture[field], /^[a-f0-9]{64}$/);
  }
  const count = data.example.tokens.length;
  assert.ok(count > 1, 'A nonzero position is needed to demonstrate rotation');
  assert.deepEqual(data.positions, Array.from({ length: count }, (_, index) => index));
  data.example.tokens.forEach((token, index) => {
    assert.equal(token.index, index);
    assert.ok(Number.isInteger(token.id) && token.id >= 0);
    assert.equal(typeof token.text, 'string');
  });
  for (const field of ['queryBefore', 'queryAfter', 'keyBefore', 'keyAfter', 'value']) {
    assert.equal(data[field].length, count, `${field}: token count`);
    data[field].forEach(row => vector(row, 128, field));
  }
  for (const field of ['cos', 'sin']) {
    assert.equal(data.rotary[field].length, count, `${field}: token count`);
    data.rotary[field].forEach(row => vector(row, 64, field));
  }
  let pairsChecked = 0;
  let maxRotationError = 0;
  for (let token = 0; token < count; token++) {
    for (const kind of ['query', 'key']) {
      const before = data[`${kind}Before`][token];
      const after = data[`${kind}After`][token];
      if (token === 0) assert.deepEqual(after, before, 'Position zero leaves the normalized head unchanged');
      for (let pair = 0; pair < 64; pair++) {
        const cosine = data.rotary.cos[token][pair];
        const sine = data.rotary.sin[token][pair];
        const angle = data.positions[token] / Math.pow(data.model.ropeTheta, 2 * pair / 128);
        near(cosine, Math.cos(angle), '128D cosine', 2e-6);
        near(sine, Math.sin(angle), '128D sine', 2e-6);
        const reconstructed = [before[pair] * cosine - before[pair + 64] * sine,
          before[pair + 64] * cosine + before[pair] * sine];
        for (let offset = 0; offset < 2; offset++) {
          const actual = after[pair + offset * 64];
          near(actual, reconstructed[offset], '128D half-split RoPE coordinate');
          maxRotationError = Math.max(maxRotationError, Math.abs(actual - reconstructed[offset]));
        }
        pairsChecked++;
      }
      near(Math.hypot(...after), Math.hypot(...before), '128D RoPE preserves head norm');
    }
  }
  return { tokens: count, headSize: 128, pairsChecked, maxRotationError };
}

export function verifyRopeExampleProvenance(data, sourceBytes) {
  assert.equal(data.capture.scriptSha256, createHash('sha256').update(sourceBytes).digest('hex'), '128D capture source hash');
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const root = process.argv[2] || 'static/traces';
  const mainBytes = readFileSync(`${root}/granite-moe.json`);
  const main = JSON.parse(mainBytes);
  const sourceBytes = readFileSync('scripts/capture_attention_operations.py');
  for (const example of main.examples) {
    const sidecar = JSON.parse(readFileSync(`${root}/attention-operations-${example.id}.json`, 'utf8'));
    verifyAttentionOperationsProvenance(sidecar, mainBytes, sourceBytes);
    console.log(example.id, JSON.stringify(validateAttentionOperations(sidecar, main)));
  }
  const ropeExample = JSON.parse(readFileSync(`${root}/rope-example-128.json`, 'utf8'));
  verifyRopeExampleProvenance(ropeExample, readFileSync(ropeExample.capture.script));
  console.log(ropeExample.example.id, JSON.stringify(validateRopeExample(ropeExample)));
}
