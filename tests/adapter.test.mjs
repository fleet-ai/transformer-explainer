import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

// Compile the production adapter without loading browser-only upstream UI modules.
// Only store plumbing is substituted; every probability and tensor operation is real.
function writable(value) {
  return {
    set(next) { value = next; },
    update(fn) { value = fn(value); },
    subscribe(fn) { fn(value); return () => {}; },
    get value() { return value; }
  };
}
const storeNames = ['attentionHeadIdx', 'attentionHeadIdxTemp', 'blockIdx',
  'expandedBlock', 'highlightedToken', 'hoveredMatrixCell', 'inputText', 'isFetchingModel',
  'isLoaded', 'isModelRunning', 'isTextbookOpen', 'modelData', 'moeDocument', 'moeExample',
  'moeModel', 'moeTokenIndex', 'predictedToken', 'sampling', 'selectedExampleIdx',
  'selectedModel', 'temperature', 'tokenIds', 'tokens'];
const stores = Object.fromEntries(storeNames.map(name => [name, writable(undefined)]));
stores.modelMetaMap = {};
stores.temperature.set(1);
stores.sampling.set({ type: 'top-p', value: 1 });
const compiled = ts.transpileModule(
  readFileSync(new URL('../src/lib/moe/adapter.ts', import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }
).outputText;
const exports = {};
runInNewContext(compiled, {
  exports,
  require(specifier) {
    if (specifier === '~/store') return stores;
    if (specifier === 'svelte/store') return { get: store => store.value };
    throw new Error(`Unexpected adapter dependency: ${specifier}`);
  }
});
const { recordedProbabilities, toOriginalModelData, hydrateRecordedExample, applyRecordedSampling } = exports;
const recorded = JSON.parse(readFileSync(new URL('../static/traces/granite-moe.json', import.meta.url), 'utf8'));
const near = (actual, expected, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);

test('displaying fewer candidates preserves full-vocabulary probability mass', () => {
  const probabilities = recordedProbabilities([0, 0, 0, 0], ['a', 'b', 'c', 'd'], 1, { type: 'top-p', value: 1 }, 2);
  assert.equal(probabilities.length, 2);
  near(probabilities[0].probability, 0.25);
  near(probabilities[1].probability, 0.25);
});

test('top-k and nucleus filtering renormalize only the retained vocabulary', () => {
  const logits = [4, 3, 2, 1].map(Math.log);
  for (const sampling of [{ type: 'top-k', value: 2 }, { type: 'top-p', value: 0.65 }]) {
    const probabilities = recordedProbabilities(logits, ['a', 'b', 'c', 'd'], 1, sampling);
    near(probabilities[0].probability, 4 / 7);
    near(probabilities[1].probability, 3 / 7);
    assert.equal(probabilities[2].probability, 0);
    assert.equal(probabilities[0].cutoffIndex, 1);
    near(probabilities[1].cumulativeProbability, 0.7);
  }
});

test('temperature rescales captured logits before filtering', () => {
  const probabilities = recordedProbabilities([Math.log(4), 0], ['a', 'b'], 2, { type: 'top-p', value: 1 });
  near(probabilities[0].probability, 2 / 3);
  near(probabilities[1].probability, 1 / 3);
  assert.throws(() => recordedProbabilities([0], ['a'], 0, { type: 'top-p', value: 1 }), /Temperature/);
});

test('top-p one retains tiny vocabulary probabilities after cumulative mass rounds to one', () => {
  const probabilities = recordedProbabilities([0, -40], ['a', 'b'], 1, { type: 'top-p', value: 1 });
  assert.equal(probabilities[0].cutoffIndex, 1);
  assert.ok(probabilities[1].probability > 0);
});

test('real captures populate original attention slots without inventing activations', () => {
  const example = recorded.examples[0];
  const adapted = toOriginalModelData(example, recorded.vocabulary);
  assert.equal(adapted.probabilities.length, 16, 'The original diagram displays sixteen vocabulary candidates');
  assert.equal(Object.keys(adapted.outputs).length, recorded.model.numLayers * recorded.model.attentionHeads * 5);
  for (const [layer, heads] of example.attention.entries()) {
    for (const [head, probabilities] of heads.entries()) {
      const prefix = `block_${layer}_attn_head_${head}`;
      const raw = adapted.outputs[`${prefix}_attn`].data;
      raw.forEach((row, query) => row.forEach((value, key) => {
        near(value * recorded.model.attentionMultiplier, example.attentionScores[layer][head][query][key]);
      }));
      assert.equal(adapted.outputs[`${prefix}_attn_scaled`].data, example.attentionScores[layer][head]);
      assert.equal(adapted.outputs[`${prefix}_attn_dropout`].data, probabilities);
      const masked = adapted.outputs[`${prefix}_attn_masked`].data;
      masked.forEach((row, query) => row.forEach((value, key) => {
        assert.equal(value, key > query ? -Infinity : example.attentionMaskedScores[layer][head][query][key]);
      }));
    }
  }
  for (const candidate of example.nextTokenCandidates) {
    const displayed = adapted.probabilities.find(item => item.tokenId === candidate.id);
    assert.ok(displayed);
    // Capture used PyTorch float32; the browser recomputes in float64.
    near(displayed.probability, candidate.probability, 5e-5);
  }
});

test('dot-product and scaling stages use the configured attention multiplier', () => {
  const example = recorded.examples[0];
  const adapted = toOriginalModelData(example, recorded.vocabulary, 1, { type: 'top-p', value: 1 }, 0.25);
  const scaled = example.attentionScores[0][0][0][0];
  assert.equal(adapted.outputs.block_0_attn_head_0_attn.data[0][0], scaled * 4);
  assert.equal(adapted.outputs.block_0_attn_head_0_attn_scaled.data[0][0], scaled);
  assert.throws(() => toOriginalModelData(example, recorded.vocabulary, 1, { type: 'top-p', value: 1 }, 0), /attention multiplier/);
});

test('hydration and controls use one recording with correct token and model alignment', () => {
  hydrateRecordedExample(recorded, 1);
  assert.equal(stores.moeDocument.value, recorded);
  assert.equal(stores.moeExample.value, recorded.examples[1]);
  assert.deepEqual(stores.tokenIds.value, recorded.examples[1].tokens.map(token => token.id));
  assert.equal(stores.moeTokenIndex.value, recorded.examples[1].tokens.length - 1);
  assert.equal(stores.modelMetaMap['granite-moe'].attention_head_num, 16);
  assert.equal(stores.isTextbookOpen.value, false);
  stores.sampling.set({ type: 'top-k', value: 1 });
  applyRecordedSampling();
  assert.equal(stores.modelData.value.probabilities[0].probability, 1);
  assert.equal(stores.modelData.value.probabilities[1].probability, 0);
  assert.equal(stores.moeExample.value, recorded.examples[1]);
});
