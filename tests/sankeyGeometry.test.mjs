import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const compiled = ts.transpileModule(
  readFileSync(new URL('../src/utils/sankeyGeometry.ts', import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }
).outputText;
const exports = {};
runInNewContext(compiled, { exports });
const { getRenderedEndpointPairs } = exports;

function endpoint(x, y, width = 8, height = 12, rendered = true) {
  const rect = { x, y, width, height, left: x, top: y, right: x + width, bottom: y + height };
  return {
    rect,
    getClientRects: () => rendered ? [rect] : [],
    getBoundingClientRect: () => rect
  };
}

test('visible endpoints retain their measured rectangles and original indices', () => {
  const sources = [endpoint(10, 20), endpoint(10, 40), endpoint(10, 60)];
  const targets = [endpoint(200, 25), endpoint(200, 45), endpoint(200, 65)];
  const pairs = getRenderedEndpointPairs(sources, targets);
  assert.equal(pairs.length, 3);
  pairs.forEach((pair, index) => {
    assert.equal(pair.index, index);
    assert.strictEqual(pair.source, sources[index].rect);
    assert.strictEqual(pair.target, targets[index].rect);
  });
});

test('hidden middle endpoints are skipped without re-pairing later tokens', () => {
  for (const hiddenSide of ['source', 'target']) {
    const sources = [endpoint(10, 20), endpoint(10, 40, 8, 12, hiddenSide !== 'source'), endpoint(10, 60)];
    const targets = [endpoint(200, 25), endpoint(200, 45, 8, 12, hiddenSide !== 'target'), endpoint(200, 65)];
    const pairs = getRenderedEndpointPairs(sources, targets);
    assert.deepEqual(Array.from(pairs, pair => pair.index), [0, 2]);
    assert.strictEqual(pairs[1].source, sources[2].rect);
    assert.strictEqual(pairs[1].target, targets[2].rect);
  }
});

test('missing targets are skipped while later original pairs remain aligned', () => {
  const sources = [endpoint(10, 20), endpoint(10, 40), endpoint(10, 60), endpoint(10, 80)];
  const targets = [endpoint(200, 25), undefined, endpoint(200, 65)];
  const pairs = getRenderedEndpointPairs(sources, targets);
  assert.deepEqual(Array.from(pairs, pair => pair.index), [0, 2]);
  assert.strictEqual(pairs[1].target, targets[2].rect);
  assert.equal(getRenderedEndpointPairs(sources, []).length, 0);
});

test('collapsed pairs and zero-height endpoints produce no connectors even when client rects exist', () => {
  const visibleSource = endpoint(10, 20);
  const visibleTarget = endpoint(200, 25);
  for (const [width, height] of [[8, 0], [0, 0]]) {
    const collapsed = endpoint(100, 100, width, height);
    assert.equal(getRenderedEndpointPairs([collapsed], [visibleTarget]).length, 0);
    assert.equal(getRenderedEndpointPairs([visibleSource], [collapsed]).length, 0);
  }
  assert.equal(getRenderedEndpointPairs([endpoint(10, 20, 0, 12)], [endpoint(200, 25, 0, 12)]).length, 0);
  const tiny = endpoint(10, 20, 0.01, 0.01);
  assert.equal(getRenderedEndpointPairs([tiny], [visibleTarget]).length, 1);
});

test('a zero-width vertical probability anchor retains its connector', () => {
  const source = endpoint(10, 20, 12, 30);
  const target = endpoint(200, 25, 0, 80);
  const pairs = getRenderedEndpointPairs([source], [target]);
  assert.equal(pairs.length, 1);
  assert.strictEqual(pairs[0].source, source.rect);
  assert.strictEqual(pairs[0].target, target.rect);
  assert.equal(pairs[0].index, 0);
  assert.equal(getRenderedEndpointPairs([target], [source]).length, 1);
});
