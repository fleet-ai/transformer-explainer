import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import * as svelte from 'svelte/store';
import ts from 'typescript';

const exports = {};
const compiled = ts.transpileModule(readFileSync(new URL('../src/lib/inspectors.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
}).outputText;
runInNewContext(compiled, { exports, require: name => {
  assert.equal(name, 'svelte/store');
  return svelte;
}});

test('switching between projection, head split, RoPE and architecture panels never overlaps', () => {
  const create = exports.createInspectorGroup();
  const projection = create(), attention = create(), architecture = create();
  const stores = [projection, attention, architecture];
  const history = [];
  const stops = stores.map(store => store.subscribe(() => history.push(stores.map(svelte.get))));
  projection.set('qkv');
  attention.set({ kind: 'heads' });
  assert.equal(svelte.get(projection), null);
  architecture.set('rope');
  assert.equal(svelte.get(attention), null);
  attention.set({ kind: 'mla', stage: 'query' });
  projection.set('attention');
  assert.equal(svelte.get(attention), null);
  assert.equal(svelte.get(architecture), null);
  assert.ok(history.every(snapshot => snapshot.filter(value => value !== null).length <= 1));
  stops.forEach(stop => stop());
});

test('closing an old panel cannot dismiss a newer inspector, including update transitions', () => {
  const create = exports.createInspectorGroup();
  const old = create(), current = create();
  old.set('qkv');
  current.set('rope');
  old.set(null);
  assert.equal(svelte.get(current), 'rope');
  old.update(value => { assert.equal(value, null); return 'qkv'; });
  assert.equal(svelte.get(current), null);
  assert.equal(svelte.get(old), 'qkv');
  old.set(null);
  current.set('attention');
  assert.equal(svelte.get(current), 'attention');
});
