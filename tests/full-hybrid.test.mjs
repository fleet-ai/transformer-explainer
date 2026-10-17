import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const root = new URL('../static/traces/hybrid/', import.meta.url);
const doc = JSON.parse(readFileSync(new URL('manifest.json', root)));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const near = (a, b, epsilon = 2e-5) => assert.ok(Math.abs(a - b) <= epsilon * Math.max(1, Math.abs(b)), `${a} != ${b}`);
function read(descriptor) {
  const bytes = readFileSync(new URL(descriptor.file, root));
  assert.equal(bytes.byteLength, descriptor.bytes);
  assert.equal(hash(bytes), descriptor.sha256);
  const floats = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  return Object.fromEntries(Object.entries(descriptor.tensors).map(([name, spec]) => {
    assert.equal(spec.shape.reduce((a, b) => a * b, 1), spec.length);
    assert.ok(spec.offset + spec.length <= floats.length);
    const values = floats.subarray(spec.offset, spec.offset + spec.length);
    assert.ok(values.every(Number.isFinite), `${descriptor.file} ${name}`);
    return [name, values];
  }));
}

test('Zamba2 records the complete 38-block pretrained model and tied vocabulary output', () => {
  assert.equal(doc.kind, 'hybrid-full');
  assert.equal(doc.capture.partial, false);
  assert.equal(doc.layers.length, 38);
  assert.equal(doc.model.numLayers, 38);
  assert.deepEqual(doc.layers.filter(x => x.type === 'hybrid').map(x => x.index), [5, 11, 17, 23, 29, 35]);
  assert.equal(doc.capture.captureScriptSha256, hash(readFileSync(new URL('../scripts/capture-hybrid-full.py', import.meta.url))));
  assert.equal(doc.validation.hookedVsUnhookedMaxError, 0);
  assert.equal(doc.capture.revision, '6b05bf29d1bb4ca71a36d12f7da4d3120dcde7fe');
  const common = read(doc.common);
  assert.equal(common.logits.length, 32000);
  assert.equal(doc.vocabulary.length, 32000);
  let best = 0;
  for (let i = 1; i < 32000; i++) if (common.logits[i] > common.logits[best]) best = i;
  assert.equal(doc.nextTokenId, best);
  assert.equal(doc.nextToken, doc.vocabulary[best]);
  assert.equal(doc.prompt + doc.nextToken, 'The capital of France is Paris');
  const last = (doc.sequenceLength - 1) * 2048;
  for (let row = 0; row < 4; row++) {
    let logit = 0;
    for (let d = 0; d < 2048; d++) logit += common.classifierWeights[row * 2048 + d] * common.finalHidden[last + d];
    near(common.logits[common.classifierTokenIds[row]], logit);
  }
});

test('every block preserves continuity and reconstructs all 64 selective-state heads', () => {
  let previous = read(doc.common).embedding;
  const n = doc.sequenceLength;
  for (const descriptor of doc.layers) {
    const t = read(descriptor);
    assert.deepEqual(t.input, previous);
    previous = t.output.slice();
    assert.equal(t.state.length, n * 64 * 4 * 128);
    for (let token = 0; token < n; token++) {
      for (let d = 0; d < 2048; d++) near(t.output[token * 2048 + d], t.input[token * 2048 + d] + t.mambaOutput[token * 2048 + d], 2e-6);
      for (let head = 0; head < 64; head++) {
        const dt = t.dt[token * 64 + head], decay = Math.exp(dt * t.A[head]);
        near(dt, Math.max(0.001, Math.log1p(Math.exp(t.dtRaw[token * 64 + head] + t.dtBias[head]))), 2e-6);
        for (let channel = 0; channel < 4; channel++) {
          const x = t.x[(token * 64 + head) * 64 + channel];
          let readout = t.D[head] * x;
          for (let state = 0; state < 128; state++) {
            const offset = ((token * 64 + head) * 4 + channel) * 128 + state;
            const old = token ? t.state[offset - 64 * 4 * 128] : 0;
            near(t.state[offset], old * decay + dt * t.B[token * 128 + state] * x, 3e-6);
            readout += t.state[offset] * t.C[token * 128 + state];
          }
          near(t.sequentialReadout[(token * 64 + head) * 64 + channel], readout, 3e-5);
          near(t.scanOutput[(token * 64 + head) * 64 + channel], readout, 5e-4);
        }
      }
    }
  }
});

test('all six shared Transformer calls use real adapted attention and feed-forward outputs', () => {
  const n = doc.sequenceLength;
  for (const descriptor of doc.layers.filter(x => x.type === 'hybrid')) {
    const t = read(descriptor);
    assert.equal(t.query.length, 32 * n * 128);
    for (let head = 0; head < 32; head++) for (let query = 0; query < n; query++) {
      const start = (head * n + query) * n;
      const scores = [];
      for (let key = 0; key < n; key++) {
        let dot = 0;
        for (let d = 0; d < 128; d++) dot += t.query[(head * n + query) * 128 + d] * t.key[(head * n + key) * 128 + d];
        near(t.scores[start + key], dot, 3e-4);
        scores.push(key > query ? -Infinity : dot / 8);
      }
      const max = Math.max(...scores), denominator = scores.reduce((sum, x) => sum + Math.exp(x - max), 0);
      scores.forEach((x, key) => near(t.attention[start + key], Math.exp(x - max) / denominator, 2e-6));
      for (const d of [0, 63, 127]) {
        let value = 0;
        for (let key = 0; key < n; key++) value += t.attention[start + key] * t.value[(head * n + key) * 128 + d];
        near(t.context[(head * n + query) * 128 + d], value);
      }
    }
    for (let token = 0; token < n; token++) {
      for (let d = 0; d < 8192; d++) near(t.product[token * 8192 + d], t.gelu[token * 8192 + d] * t.up[token * 8192 + d], 2e-6);
      // The shared Transformer contribution enters Mamba's normalization, not a separate residual output.
      const hidden = Array.from({ length: 2048 }, (_, d) => t.input[token * 2048 + d] + t.transformerOutput[token * 2048 + d]);
      const rms = Math.sqrt(hidden.reduce((sum, v) => sum + v * v, 0) / 2048 + 1e-5);
      for (let d = 0; d < 2048; d++) near(t.norm1[token * 2048 + d], hidden[d] / rms * t.normWeight[d], 2e-5);
    }
  }
});
