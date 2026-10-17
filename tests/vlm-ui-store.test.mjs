import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import * as svelte from 'svelte/store';
import ts from 'typescript';

// Exercise the production derived stores with Svelte's real notification ordering
// and captured tensors; only the browser app's block selector is supplied here.
function loadModule(path, dependencies) {
  const compiled = ts.transpileModule(readFileSync(new URL(path, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText;
  const exports = {};
  runInNewContext(compiled, { exports, require: name => {
    assert.ok(name in dependencies, `Unexpected dependency ${name}`);
    return dependencies[name];
  }});
  return exports;
}
const tensors = loadModule('../src/lib/vit/types.ts', {});
const blockIdx = svelte.writable(0);
const stores = loadModule('../src/lib/vlm/store.ts', {
  'svelte/store': svelte, '~/store': { blockIdx }, '$lib/vit/types': tensors
});
const root = new URL('../static/traces/vlm/', import.meta.url);
const document = JSON.parse(readFileSync(new URL('manifest.json', root)));
function layer(phase) {
  const descriptor = document.streams[phase].layers[0];
  const bytes = readFileSync(new URL(`${phase}/${descriptor.file}`, root));
  return { descriptor, values: new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4) };
}

test('changing VLM streams never reads the other stream’s tensors, including during lazy loading', () => {
  const vision = layer('vision'), language = layer('language');
  const recording = { document, layers: { vision: new Map([[0, vision]]), language: new Map() } };
  let shown;
  const stop = stores.vlmHeadOperations.subscribe(value => { shown = value; });
  stores.vlmRecording.set(recording);
  assert.equal(shown.queryBefore.length, 12);
  assert.equal(shown.queryBefore[0][0][0], tensors.scalar(vision, 'query', 0, 0, 0));

  assert.doesNotThrow(() => stores.vlmPhase.set('language'));
  assert.equal(shown, undefined, 'The language block is still loading');
  recording.layers.language.set(0, language);
  stores.vlmRecording.set(recording);
  assert.equal(shown.queryBefore.length, 9);
  assert.equal(shown.keyBefore.length, 3);
  assert.equal(shown.queryBefore[8][5][31], tensors.scalar(language, 'queryBefore', 8, 5, 31));

  stores.vlmPhase.set('vision');
  stores.vlmVisiblePositions.set([248, 249, 250, 251, 252, 253, 254, 255]);
  assert.equal(shown.queryBefore[11][7][63], tensors.scalar(vision, 'query', 11, 255, 63));
  assert.doesNotThrow(() => stores.vlmPhase.set('language'));
  assert.equal(shown, undefined, 'Previous image positions cannot address the shorter language sequence');
  stores.vlmVisiblePositions.set([30, 31, 32, 33, 34, 35, 36, 37]);
  assert.equal(shown.queryBefore[8][7][63], tensors.scalar(language, 'queryBefore', 8, 37, 63));
  blockIdx.set(1);
  assert.equal(shown, undefined, 'A newly selected uncached block does not retain stale head vectors');
  stop();
});

test('hydrating a lazy block preserves current selectors without queuing zero-block reloads', () => {
  const names = ['attentionHeadIdx', 'attentionHeadIdxTemp', 'blockIdx', 'expandedBlock',
    'highlightedToken', 'hoveredMatrixCell', 'inputText', 'isFetchingModel', 'isLoaded',
    'isModelRunning', 'isTextbookOpen', 'modelData', 'moeDocument', 'moeExample', 'moeModel',
    'moeTokenIndex', 'moeAttentionOperations', 'predictedToken', 'sampling', 'selectedExampleIdx',
    'selectedModel', 'temperature', 'tokenIds', 'tokens'];
  const app = Object.fromEntries(names.map(name => [name, svelte.writable(undefined)]));
  app.modelMetaMap = {};
  app.temperature.set(1);
  app.sampling.set({ type: 'top-p', value: 1 });
  app.blockIdx.set(5);
  app.attentionHeadIdx.set(8);
  app.attentionHeadIdxTemp.set(8);
  const adapter = loadModule('../src/lib/moe/adapter.ts', { '~/store': app, 'svelte/store': svelte });
  const trace = JSON.parse(readFileSync(new URL('../static/traces/granite-moe.json', import.meta.url)));
  const seen = [];
  const stop = app.blockIdx.subscribe(value => { seen.push(value); });
  adapter.hydrateRecordedExample(trace, 0, { preserveSelection: true });
  assert.deepEqual(seen, [5], 'No intermediate block 0 is emitted to lazy loaders');
  assert.equal(svelte.get(app.attentionHeadIdx), 8);
  assert.equal(svelte.get(app.attentionHeadIdxTemp), 8);
  adapter.hydrateRecordedExample(trace);
  assert.deepEqual(seen, [5, 0], 'Changing the example still resets selectors by default');
  assert.equal(svelte.get(app.attentionHeadIdx), 0);
  stop();
});
