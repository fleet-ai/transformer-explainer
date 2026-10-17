import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { validateTraces } from '../scripts/validate-traces.mjs';

const recorded = JSON.parse(readFileSync(new URL('../static/traces/granite-moe.json', import.meta.url), 'utf8'));
test('real captures preserve routing, weighted outputs, hidden previews and causal attention', () => {
  const result = validateTraces(recorded);
  assert.ok(result.tokenLayerPairs > 100);
  assert.equal(result.examples, 3);
  assert.equal(recorded.model.numLayers, 24);
  assert.equal(recorded.model.attentionHeads, 16);
  assert.equal(result.attentionRows, result.tokenLayerPairs * recorded.model.attentionHeads);
});
test('rejects attention that leaks to a future token even when its probability mass is preserved', () => {
  const data = structuredClone(recorded);
  const row = data.examples[0].attention[0][0][0];
  row[0] -= 0.1;
  row[1] = 0.1;
  assert.throws(() => validateTraces(data), /Future-token attention/);
});
test('rejects attention rows whose probabilities do not sum to one', () => {
  const data = structuredClone(recorded);
  data.examples[0].attention[0][0][0][0] = 0.9;
  assert.throws(() => validateTraces(data), /Attention row probability mass/);
});
test('rejects malformed layer, head, query and key attention dimensions', () => {
  for (const [dimension, label] of ['layers', 'heads', 'queries', 'keys'].entries()) {
    const data = structuredClone(recorded);
    let array = data.examples[0].attention;
    for (let depth = 0; depth < dimension; depth++) array = array[0];
    array.pop();
    assert.throws(() => validateTraces(data), new RegExp(`Attention ${label} length`));
  }
});
test('rejects non-finite or out-of-range attention probabilities', () => {
  for (const value of [NaN, Infinity, -0.1, 1.1]) {
    const data = structuredClone(recorded);
    data.examples[0].attention[0][0][0][0] = value;
    assert.throws(() => validateTraces(data), /Attention (keys contains non-finite values|probabilities out of range)/);
  }
});
test('requires eight-dimensional embedding, final-hidden and attention previews with valid norms', () => {
  for (const [prefix, label, atLayer] of [
    ['embedding', 'Embedding', false],
    ['finalHidden', 'Final hidden', false],
    ['attentionInput', 'Attention input', true],
    ['attentionOutput', 'Attention output', true],
  ]) {
    for (const corruption of ['missing', 'coordinate', 'norm', 'negative norm']) {
      const data = structuredClone(recorded);
      const owner = atLayer ? data.examples[0].layers[0] : data.examples[0];
      if (corruption === 'missing') delete owner[`${prefix}Preview`];
      if (corruption === 'coordinate') owner[`${prefix}Preview`][0].pop();
      if (corruption === 'norm') owner[`${prefix}Norms`][0] = Infinity;
      if (corruption === 'negative norm') owner[`${prefix}Norms`][0] = -1;
      assert.throws(() => validateTraces(data), new RegExp(label));
    }
  }
});
test('rejects a routing weight that is not the actual top-k softmax', () => {
  const data = structuredClone(recorded);
  data.examples[0].layers[0].tokens[0].routingWeights[0] += 0.05;
  assert.throws(() => validateTraces(data), /Selected weight/);
});
test('rejects an expert output that does not combine to the recorded MoE output', () => {
  const data = structuredClone(recorded);
  data.examples[0].layers[0].tokens[0].combinedOutputPreview[0] += 1;
  assert.throws(() => validateTraces(data), /Weighted sum coordinate/);
});
test('rejects a trace with missing token rows', () => {
  const data = structuredClone(recorded);
  data.examples[1].layers[1].tokens.pop();
  assert.throws(() => validateTraces(data), /Token-layer alignment/);
});
test('rejects expert load counts that do not match routed tokens', () => {
  const data = structuredClone(recorded);
  const layer = data.examples[0].layers[0];
  layer.expertTokenCounts ||= Array(data.model.numExperts).fill(0);
  layer.expertTokenCounts[0] += 1;
  assert.throws(() => validateTraces(data), /Expert load accounting/);
});
test('downloadable capture source matches recorded SHA-256 provenance', async () => {
  const { createHash } = await import('node:crypto');
  const script = readFileSync(new URL('../scripts/capture_moe_traces.py', import.meta.url));
  const downloaded = readFileSync(new URL('../static/traces/capture_moe_traces.py', import.meta.url));
  assert.equal(createHash('sha256').update(script).digest('hex'), recorded.capture.scriptSha256);
  assert.ok(downloaded.equals(script), 'Downloadable capture script differs from provenance source');
  assert.ok(
    readFileSync(new URL('../static/traces/requirements-traces.txt', import.meta.url)).equals(
      readFileSync(new URL('../scripts/requirements-traces.txt', import.meta.url))
    ), 'Downloadable dependency pins differ from source'
  );
  for (const example of recorded.examples) assert.equal(example.validation.hookedVsUnhookedMaxLogitError, 0);
});
