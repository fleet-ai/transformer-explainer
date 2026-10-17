import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const root = new URL('../static/traces/vlm/', import.meta.url);
const doc = JSON.parse(readFileSync(new URL('manifest.json', root)));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const near = (actual, expected, tolerance = 1e-5) => assert.ok(
  Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);
function read(phase, descriptor) {
  const bytes = readFileSync(new URL(`${phase ? phase + '/' : ''}${descriptor.file}`, root));
  assert.equal(bytes.byteLength, descriptor.bytes);
  assert.equal(hash(bytes), descriptor.sha256);
  const floats = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const tensors = {};
  for (const [name, spec] of Object.entries(descriptor.tensors)) {
    assert.equal(spec.shape.reduce((a, b) => a * b, 1), spec.length);
    assert.ok(spec.offset + spec.length <= floats.length);
    const values = floats.subarray(spec.offset, spec.offset + spec.length);
    assert.ok(values.every(Number.isFinite), `${phase} ${descriptor.file} ${name}`);
    tensors[name] = values;
  }
  return tensors;
}

test('SmolVLM executes all vision and language blocks with unchanged pretrained logits', () => {
  assert.equal(doc.kind, 'vlm-full');
  assert.equal(doc.streams.vision.layers.length, 12);
  assert.equal(doc.streams.language.layers.length, 30);
  assert.equal(doc.streams.vision.model.attentionHeads, 12);
  assert.equal(doc.streams.language.model.attentionHeads, 9);
  assert.equal(doc.streams.language.model.keyValueHeads, 3);
  assert.equal(doc.image.inputSize, 256);
  assert.equal(doc.streams.vision.sequenceLength, 256);
  assert.equal(doc.imageTokenPositions.length, 16);
  assert.equal(doc.validation.hookedVsUnhookedLogitMaxError, 0);
  assert.equal(doc.capture.scriptSha256, hash(readFileSync(new URL('../scripts/capture-vlm-full.py', import.meta.url))));
  assert.equal(doc.capture.weightsSha256, '74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e');
  const language = read('language', doc.streams.language.common);
  assert.equal(language.logits.length, doc.vocabulary.length);
  let best = 0;
  for (let i = 1; i < language.logits.length; i++) if (language.logits[i] > language.logits[best]) best = i;
  assert.equal(best, doc.completionIds[0], 'Greedy generation starts from the recorded full-vocabulary argmax');
});

test('all 42 recorded blocks reconstruct real attention and both residual additions', () => {
  for (const [phase, stream] of Object.entries(doc.streams)) {
    const n = stream.sequenceLength, heads = stream.model.attentionHeads, kvHeads = stream.model.keyValueHeads;
    const width = stream.model.hiddenSize, middle = stream.model.intermediateSize;
    let previousOutput;
    for (const descriptor of stream.layers) {
      const values = read(phase, descriptor);
      assert.equal(values.query.length, heads * n * 64);
      assert.equal(values.key.length, kvHeads * n * 64);
      assert.equal(values.value.length, kvHeads * n * 64);
      assert.equal(values.attention.length, heads * n * n);
      if (previousOutput) assert.deepEqual(values.input, previousOutput, 'Every block consumes its predecessor’s exact output');
      previousOutput = values.output.slice();
      for (let head = 0; head < heads; head++) {
        const kvHead = Math.floor(head / (heads / kvHeads));
        for (let query = 0; query < n; query++) {
          const start = (head * n + query) * n;
          near(values.attention.subarray(start, start + n).reduce((a, b) => a + b, 0), 1, 1e-6);
          if (phase === 'language') for (let key = query + 1; key < n; key++) assert.equal(values.attention[start + key], 0);
        }
        for (const query of [0, Math.floor(n / 2), n - 1]) {
          const start = (head * n + query) * n;
          let max = -Infinity;
          const scores = [];
          for (let key = 0; key < n; key++) {
            let dot = 0;
            for (let d = 0; d < 64; d++) dot += values.query[(head * n + query) * 64 + d] * values.key[(kvHead * n + key) * 64 + d];
            near(values.scores[start + key], dot, 2e-4);
            const score = phase === 'language' && key > query ? -Infinity : dot / 8;
            scores.push(score); max = Math.max(max, score);
          }
          const denominator = scores.reduce((sum, score) => sum + Math.exp(score - max), 0);
          scores.forEach((score, key) => near(values.attention[start + key], Math.exp(score - max) / denominator, 2e-6));
          for (const d of [0, 17, 63]) {
            let sum = 0;
            for (let key = 0; key < n; key++) sum += values.attention[start + key] * values.value[(kvHead * n + key) * 64 + d];
            near(values.context[(head * n + query) * 64 + d], sum, 2e-5);
          }
        }
      }
      for (let token = 0; token < n; token++) for (let coordinate = 0; coordinate < 4; coordinate++) {
        const i = token * width + coordinate;
        near(values.residualAfterAttention[i], values.input[i] + values.attentionOutput[i], 2e-6);
        near(values.output[i], values.residualAfterAttention[i] + values.down[i], 2e-6);
        if (phase === 'language') {
          const d = token * middle + coordinate, gate = values.gate[d];
          near(values.siluGate[d], gate / (1 + Math.exp(-gate)), 2e-6);
          near(values.product[d], values.siluGate[d] * values.up[d], 2e-6);
        }
      }
    }
  }
});

test('the complete image connector produces the exact vectors merged into language input', () => {
  const connector = read('', doc.connector);
  const language = read('language', doc.streams.language.common);
  const vision = read('vision', doc.streams.vision.common);
  const visionWidth = doc.streams.vision.model.hiddenSize;
  const languageWidth = doc.streams.language.model.hiddenSize;
  const packedWidth = visionWidth * 16;
  for (let group = 0; group < 16; group++) {
    const groupRow = Math.floor(group / 4), groupColumn = group % 4;
    for (let dy = 0; dy < 4; dy++) for (let dx = 0; dx < 4; dx++) for (let d = 0; d < visionWidth; d++) {
      const patch = (groupRow * 4 + dy) * 16 + groupColumn * 4 + dx;
      assert.equal(connector.packed[group * packedWidth + (dy * 4 + dx) * visionWidth + d], vision.finalHidden[patch * visionWidth + d]);
    }
    for (let d = 0; d < 4; d++) {
      let sum = 0;
      for (let x = 0; x < packedWidth; x++) sum += connector.packed[group * packedWidth + x] * connector.weights[d * packedWidth + x];
      near(connector.projected[group * languageWidth + d], sum, 2e-4);
    }
    for (let d = 0; d < languageWidth; d++) assert.equal(connector.projected[group * languageWidth + d], language.embedding[doc.imageTokenPositions[group] * languageWidth + d]);
  }
});
