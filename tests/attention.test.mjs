import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';
import { validateAttentionOperations, verifyAttentionOperationsProvenance, validateRopeExample, verifyRopeExampleProvenance } from '../scripts/validate-attention-operations.mjs';

const root = new URL('../static/traces/', import.meta.url);
const mainBytes = readFileSync(new URL('granite-moe.json', root));
const main = JSON.parse(mainBytes);
const sidecars = main.examples.map(example => JSON.parse(readFileSync(new URL(`attention-operations-${example.id}.json`, root), 'utf8')));
const ropeExample = JSON.parse(readFileSync(new URL('rope-example-128.json', root), 'utf8'));
const liveCompiled = ts.transpileModule(
  readFileSync(new URL('../src/lib/moe/live.ts', import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }
).outputText;
const compiled = ts.transpileModule(
  readFileSync(new URL('../src/lib/moe/attention.ts', import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }
).outputText;

function production(fetch = () => { throw new Error('Unexpected network request'); }) {
  const exports = {};
  const live = {};
  runInNewContext(liveCompiled, { exports: live });
  runInNewContext(compiled, {
    exports, fetch,
    require(name) {
      if (name === '$app/paths') return { base: '/moe' };
      if (name === './live') return live;
      throw new Error(`Unexpected helper dependency: ${name}`);
    }
  });
  return exports;
}
const { headMapping, ropeCoordinate, ropePair, ropePairWindow } = production();

test('real full-head captures preserve rotary arithmetic, shared K/V, and recorded attention scores', () => {
  let queryVectors = 0;
  let keyVectors = 0;
  for (const sidecar of sidecars) {
    const result = validateAttentionOperations(sidecar, main);
    queryVectors += result.queryVectors;
    keyVectors += result.keyVectors;
    assert.equal(result.valueVectors, result.keyVectors);
    assert.equal(result.rotatedPairs, (result.queryVectors + result.keyVectors) * 32);
    assert.equal(result.attentionDots, sidecar.layers.length * 16 * sidecar.example.tokens.length ** 2);
  }
  assert.equal(queryVectors, main.examples.reduce((sum, example) => sum + example.tokens.length * 24 * 16, 0));
  assert.equal(queryVectors, keyVectors * 2);
});

test('all 16 query heads map to the actual shared K/V pair without changing head size', () => {
  const sidecar = sidecars[0];
  const example = main.examples.find(item => item.id === sidecar.example.id);
  for (let head = 0; head < 16; head++) {
    const mapping = headMapping(head);
    assert.equal(mapping.queryHead, head);
    assert.equal(mapping.keyValueHead, Math.floor(head / 2));
    assert.deepEqual(Array.from(mapping.sharedQueryHeads), [Math.floor(head / 2) * 2, Math.floor(head / 2) * 2 + 1]);
    const layer = sidecar.layers[23];
    assert.equal(layer.queryAfter[head][0].length, 64);
    assert.equal(layer.keyAfter[mapping.keyValueHead][0].length, 64);
    for (const shared of mapping.sharedQueryHeads) {
      assert.deepEqual(layer.keyAfter[mapping.keyValueHead][0].slice(0, 8), example.layers[23].keyPreview[shared][0]);
      assert.deepEqual(layer.value[mapping.keyValueHead][0].slice(0, 8), example.layers[23].valuePreview[shared][0]);
    }
  }
  for (const head of [-1, 16, 0.5, NaN]) assert.throws(() => headMapping(head), /head range/);
  assert.throws(() => headMapping(0, 16, 3), /multiple/);
});

test('pair helper reconstructs all 32 half-split pairs from actual pre/post RoPE values', () => {
  for (const sidecar of sidecars) {
    for (const layerIndex of [0, 11, 23]) {
      const layer = sidecar.layers[layerIndex];
      for (const token of [0, sidecar.example.tokens.length - 1]) {
        for (const [kind, heads] of [['query', [0, 1, 14, 15]], ['key', [0, 7]]]) {
          for (const head of heads) {
            const before = layer[`${kind}Before`][head][token];
            const after = layer[`${kind}After`][head][token];
            if (token === 0) assert.deepEqual(after, before, 'Position zero leaves Q/K unchanged');
            for (let pair = 0; pair < 32; pair++) {
              const result = ropePair(before, after, sidecar.rotary.cos[token], sidecar.rotary.sin[token], pair);
              assert.deepEqual(Array.from(result.indices), [pair, pair + 32]);
              assert.deepEqual(Array.from(result.before), [before[pair], before[pair + 32]]);
              assert.deepEqual(Array.from(result.recorded), [after[pair], after[pair + 32]]);
              result.reconstructed.forEach((value, index) => {
                assert.ok(Math.abs(value - result.recorded[index]) <= 1e-5 + 2e-6 * Math.abs(result.recorded[index]));
                assert.equal(result.difference[index], value - result.recorded[index]);
              });
            }
          }
        }
      }
    }
  }
  const layer = sidecars[0].layers[0];
  const args = [layer.queryBefore[0][0], layer.queryAfter[0][0], sidecars[0].rotary.cos[0], sidecars[0].rotary.sin[0]];
  for (const pair of [-1, 32, 0.5, NaN]) assert.throws(() => ropePair(...args, pair), /pair is outside/);
  assert.throws(() => ropePair(args[0].slice(0, 63), ...args.slice(1), 0), /even-sized/);
});

test('any selected coordinate preserves the actual partner and canonical rotation signs', () => {
  for (const sidecar of sidecars) {
    const token = sidecar.example.tokens.length - 1;
    for (const layerIndex of [0, 11, 23]) {
      for (const kind of ['query', 'key']) {
        const layer = sidecar.layers[layerIndex];
        const before = layer[`${kind}Before`][0][token];
        const after = layer[`${kind}After`][0][token];
        for (let coordinate = 0; coordinate < 64; coordinate++) {
          const selected = ropeCoordinate(coordinate);
          const partner = ropeCoordinate(selected.partner);
          assert.equal(selected.coordinate, coordinate);
          assert.equal(selected.partner, coordinate < 32 ? coordinate + 32 : coordinate - 32);
          assert.equal(selected.pairIndex, coordinate % 32);
          assert.equal(partner.partner, coordinate);
          assert.equal(partner.pairIndex, selected.pairIndex);
          const result = ropePair(before, after, sidecar.rotary.cos[token], sidecar.rotary.sin[token], selected.pairIndex);
          const samePair = ropePair(before, after, sidecar.rotary.cos[token], sidecar.rotary.sin[token], partner.pairIndex);
          assert.deepEqual(result, samePair, 'Selecting the other half must not reverse rotation order or signs');
          const offset = coordinate < 32 ? 0 : 1;
          assert.equal(result.indices[offset], coordinate);
          assert.equal(result.recorded[offset], after[coordinate]);
          assert.ok(Math.abs(result.reconstructed[offset] - after[coordinate]) <= 1e-5 + 2e-6 * Math.abs(after[coordinate]));
        }
      }
    }
  }
  for (const headSize of [0, -2, 1, 63, 1.5, NaN, Infinity]) {
    assert.throws(() => ropeCoordinate(0, headSize), /head size/);
  }
  for (const coordinate of [-1, 64, 0.5, NaN, Infinity]) {
    assert.throws(() => ropeCoordinate(coordinate), /coordinate is outside/);
  }
});

test('all 128 selections reconstruct genuine Qwen Q/K pairs in canonical order', () => {
  const report = validateRopeExample(ropeExample);
  assert.equal(report.headSize, 128);
  assert.equal(report.pairsChecked, ropeExample.positions.length * 2 * 64);
  for (const [token, position] of ropeExample.positions.entries()) {
    for (const kind of ['query', 'key']) {
      const before = ropeExample[`${kind}Before`][token];
      const after = ropeExample[`${kind}After`][token];
      for (let coordinate = 0; coordinate < 128; coordinate++) {
        const selected = ropeCoordinate(coordinate, 128);
        const partner = ropeCoordinate(selected.partner, 128);
        assert.equal(selected.coordinate, coordinate);
        assert.equal(selected.partner, coordinate < 64 ? coordinate + 64 : coordinate - 64);
        assert.equal(selected.pairIndex, coordinate % 64);
        assert.equal(partner.pairIndex, selected.pairIndex);
        assert.equal(partner.partner, coordinate);
        const pair = ropePair(before, after, ropeExample.rotary.cos[token], ropeExample.rotary.sin[token], selected.pairIndex);
        const samePair = ropePair(before, after, ropeExample.rotary.cos[token], ropeExample.rotary.sin[token], partner.pairIndex);
        assert.deepEqual(pair, samePair, 'Either selected half uses the same signed rotation');
        const offset = coordinate < 64 ? 0 : 1;
        assert.equal(pair.indices[offset], coordinate);
        assert.equal(pair.recorded[offset], after[coordinate]);
        assert.ok(Math.abs(pair.reconstructed[offset] - after[coordinate]) <= 1e-5 + 2e-6 * Math.abs(after[coordinate]),
          `${kind}, position ${position}, coordinate ${coordinate}`);
      }
    }
  }
  assert.throws(() => ropeCoordinate(128, 128), /coordinate is outside/);
});

test('128D validation rejects reversed rotation signs, fake dimensions, and incorrect input normalization', () => {
  for (const mutate of [
    data => { data.queryAfter[1][0] = data.queryBefore[1][0] * data.rotary.cos[1][0] + data.queryBefore[1][64] * data.rotary.sin[1][0]; },
    data => { data.model.headSize = 64; },
    data => { data.capture.queryInputNormalization = 'none'; },
    data => { data.keyBefore[1].pop(); }
  ]) {
    const data = structuredClone(ropeExample);
    mutate(data);
    assert.throws(() => validateRopeExample(data));
  }
});

test('four-value windows preserve two distinct real pairs at every 64D and 128D coordinate', () => {
  const captures = [
    ...sidecars.flatMap(data => [0, 23].map(layer => ({
      headSize: data.model.headSize,
      queryBefore: data.layers[layer].queryBefore[0], queryAfter: data.layers[layer].queryAfter[0],
      keyBefore: data.layers[layer].keyBefore[0], keyAfter: data.layers[layer].keyAfter[0],
      rotary: data.rotary, positions: data.positions
    }))),
    { ...ropeExample, headSize: ropeExample.model.headSize }
  ];
  for (const data of captures) {
    for (const token of [0, data.positions.length - 1]) {
      for (const kind of ['query', 'key']) {
        const before = data[`${kind}Before`][token];
        const after = data[`${kind}After`][token];
        for (let coordinate = 0; coordinate < data.headSize; coordinate++) {
          const pairs = ropePairWindow(before, after, data.rotary.cos[token], data.rotary.sin[token], coordinate);
          assert.equal(pairs.length, 2);
          const indices = Array.from(pairs.flatMap(pair => pair.indices));
          assert.equal(new Set(indices).size, 4, 'Four different captured coordinates must be shown');
          assert.ok(indices.includes(coordinate), 'The selected coordinate must stay visible in either half');
          assert.ok(indices.every(index => index >= 0 && index < data.headSize));
          assert.equal(pairs[1].indices[0], pairs[0].indices[0] + 1, 'The second pair has the adjacent frequency');
          if (coordinate % (data.headSize / 2) === data.headSize / 2 - 1) {
            assert.deepEqual(indices, [data.headSize / 2 - 2, data.headSize - 2, data.headSize / 2 - 1, data.headSize - 1]);
          } else {
            assert.equal(pairs[0].indices[0], coordinate % (data.headSize / 2));
          }
          for (const pair of pairs) {
            const [first, second] = pair.indices;
            assert.equal(second - first, data.headSize / 2);
            assert.equal(pair.cosine, data.rotary.cos[token][first]);
            assert.equal(pair.sine, data.rotary.sin[token][first]);
            assert.deepEqual(Array.from(pair.before), [before[first], before[second]]);
            assert.deepEqual(Array.from(pair.recorded), [after[first], after[second]]);
            pair.reconstructed.forEach((value, offset) => {
              assert.ok(Math.abs(value - pair.recorded[offset]) <= 1e-5 + 2e-6 * Math.abs(pair.recorded[offset]));
            });
            if (token === 0) assert.deepEqual(Array.from(pair.recorded), Array.from(pair.before));
          }
        }
      }
    }
  }
  const before = ropeExample.queryBefore[4], after = ropeExample.queryAfter[4];
  const cos = ropeExample.rotary.cos[4], sin = ropeExample.rotary.sin[4];
  assert.deepEqual(Array.from(ropePairWindow(before, after, cos, sin, 123).flatMap(pair => pair.indices)), [59, 123, 60, 124]);
  for (const coordinate of [-1, 128, 0.5, NaN]) assert.throws(() => ropePairWindow(before, after, cos, sin, coordinate), /coordinate is outside/);
  assert.throws(() => ropePairWindow([1, 2], [1, 2], [1], [0], 0), /at least four/);
  assert.throws(() => ropePairWindow(before, after.slice(1), cos, sin, 0), /even-sized/);
});

test('independent 128D loader retries failures and caches the genuine example', async () => {
  const responses = [
    { ok: false, status: 503 },
    { ok: true, json: async () => ({ ...ropeExample, model: { ...ropeExample.model, headSize: 64 } }) },
    { ok: true, json: async () => ropeExample }
  ];
  const { loadRopeExample } = production(async url => {
    assert.equal(url, '/moe/traces/rope-example-128.json');
    return responses.shift();
  });
  await assert.rejects(loadRopeExample(), /HTTP 503/);
  await assert.rejects(loadRopeExample(), /128-dimensional/);
  const first = loadRopeExample();
  assert.equal(first, loadRopeExample());
  assert.equal(await first, ropeExample);
  assert.equal(await loadRopeExample(), ropeExample);
  assert.equal(responses.length, 0);
});

test('independent 128D recording retains pinned-model and exact downloadable source provenance', () => {
  const source = readFileSync(new URL(`../${ropeExample.capture.script}`, import.meta.url));
  const downloaded = readFileSync(new URL(ropeExample.capture.script.split('/').at(-1), root));
  assert.deepEqual(source, downloaded);
  verifyRopeExampleProvenance(ropeExample, source);
  assert.equal(ropeExample.model.id, 'Qwen/Qwen3-0.6B');
  assert.equal(ropeExample.capture.hookedVsUnhookedMaxLogitError, 0);
});

test('rejects changed rotation partners, modified values, wrong shared-head counts, and non-finite coordinates', () => {
  for (const mutate of [
    data => { data.layers[0].queryBefore[0][1][32] += 1; },
    data => { data.layers[0].value[0][0][0] += 1; },
    data => { data.layers[0].keyAfter.pop(); },
    data => { data.layers[0].queryAfter[0][0][63] = NaN; },
    data => { data.rotary.cos[0].pop(); }
  ]) {
    const data = structuredClone(sidecars[0]);
    mutate(data);
    assert.throws(() => validateAttentionOperations(data, main));
  }
});

test('loader shares an in-flight request and returns only a validated matching recording', async () => {
  const sidecar = sidecars[0];
  let calls = 0;
  const { loadAttentionOperations } = production(async url => {
    calls++;
    assert.equal(url, `/moe/traces/attention-operations-${sidecar.example.id}.json`);
    return { ok: true, json: async () => sidecar };
  });
  const first = loadAttentionOperations(sidecar.example.id);
  const second = loadAttentionOperations(sidecar.example.id);
  assert.equal(first, second);
  assert.equal(await first, sidecar);
  assert.equal(await loadAttentionOperations(sidecar.example.id), sidecar);
  assert.equal(calls, 1);
});

test('loader evicts HTTP, identity, and malformed-value failures so retry can succeed', async () => {
  const sidecar = sidecars[0];
  const malformed = structuredClone(sidecar);
  malformed.layers[0].queryAfter[0][0][0] = NaN;
  const responses = [
    { ok: false, status: 503 },
    { ok: true, json: async () => ({ ...sidecar, example: { ...sidecar.example, id: 'wrong-prompt' } }) },
    { ok: true, json: async () => malformed },
    { ok: true, json: async () => sidecar }
  ];
  const { loadAttentionOperations } = production(async () => responses.shift());
  await assert.rejects(loadAttentionOperations(sidecar.example.id), /HTTP 503/);
  await assert.rejects(loadAttentionOperations(sidecar.example.id), /does not match/);
  await assert.rejects(loadAttentionOperations(sidecar.example.id), /invalid queryAfter/);
  assert.equal(await loadAttentionOperations(sidecar.example.id), sidecar);
  assert.equal(responses.length, 0);
});

test('attention-operation provenance points to unchanged main bundle and exact downloadable capture source', () => {
  const source = readFileSync(new URL('../scripts/capture_attention_operations.py', import.meta.url));
  const downloaded = readFileSync(new URL('capture_attention_operations.py', root));
  assert.deepEqual(source, downloaded);
  for (const sidecar of sidecars) verifyAttentionOperationsProvenance(sidecar, mainBytes, source);
});
