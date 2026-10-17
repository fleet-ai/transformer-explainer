import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

// Execute the production TypeScript helper. Its only import is a type.
const compiled = ts.transpileModule(
  readFileSync(new URL('../src/lib/moe/arithmetic.ts', import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }
).outputText;
const exports = {};
runInNewContext(compiled, { exports });
const { sumCoordinate } = exports;
const recorded = JSON.parse(readFileSync(new URL('../static/traces/granite-moe.json', import.meta.url), 'utf8'));

test('selected expert coordinate sums reconstruct actual outputs across prompts, layers, and tokens', () => {
  let coordinates = 0;
  let signedContributions = 0;
  for (const example of recorded.examples) {
    const tokenIndices = new Set([0, Math.floor(example.tokens.length / 2), example.tokens.length - 1]);
    for (const layer of [0, 7, 15, 23]) {
      for (const token of tokenIndices) {
        const trace = example.layers[layer].tokens[token];
        for (let coordinate = 0; coordinate < 8; coordinate++) {
          const result = sumCoordinate(trace, coordinate, recorded.model.residualMultiplier);
          assert.equal(result.terms.length, recorded.model.topK);
          result.terms.forEach((term, rank) => {
            assert.equal(term.expert, trace.selectedExperts[rank]);
            assert.equal(term.weight, trace.routingWeights[rank]);
            assert.equal(term.value, trace.expertOutputPreview[rank][coordinate]);
            assert.equal(term.contribution, term.weight * term.value);
            if (term.contribution < 0) signedContributions++;
          });
          assert.equal(result.recorded, trace.combinedOutputPreview[coordinate]);
          // Match the existing trace validator's float32 reconstruction tolerance.
          const tolerance = Math.max(1e-5, Math.abs(result.recorded) * 1e-5);
          assert.ok(Math.abs(result.reconstructed - result.recorded) <= tolerance,
            `${example.id}, layer ${layer}, token ${token}, coordinate ${coordinate}`);
          assert.equal(result.difference, result.reconstructed - result.recorded);
          assert.equal(result.scaled, result.recorded * recorded.model.residualMultiplier);
          coordinates++;
        }
      }
    }
  }
  assert.equal(recorded.examples.length, 3);
  assert.ok(coordinates >= 250);
  assert.ok(signedContributions > 0, 'Vector contributions must preserve negative coordinates');
});

test('reports a signed mismatch instead of replacing the captured result with its reconstruction', () => {
  const trace = structuredClone(recorded.examples[0].layers[0].tokens[0]);
  const original = sumCoordinate(trace, 0, 0.22);
  trace.combinedOutputPreview[0] += 0.5;
  const before = structuredClone(trace);
  const result = sumCoordinate(trace, 0, 0.22);
  assert.equal(result.reconstructed, original.reconstructed);
  assert.ok(Math.abs(result.difference - (original.difference - 0.5)) < 1e-12);
  assert.equal(result.scaled, trace.combinedOutputPreview[0] * 0.22);
  assert.deepEqual(trace, before, 'Coordinate inspection must not mutate the recording');
});

test('rejects unavailable coordinates, misaligned expert rows, and non-finite arithmetic', () => {
  const trace = recorded.examples[0].layers[0].tokens[0];
  for (const coordinate of [-1, 0.5, 8, NaN, Infinity]) {
    assert.throws(() => sumCoordinate(trace, coordinate, 0.22), /captured output preview/);
  }
  for (const multiplier of [NaN, Infinity, -Infinity]) {
    assert.throws(() => sumCoordinate(trace, 0, multiplier), /multiplier must be finite/);
  }
  const misaligned = structuredClone(trace);
  misaligned.routingWeights.pop();
  assert.throws(() => sumCoordinate(misaligned, 0, 0.22), /must align/);
  const missingCoordinate = structuredClone(trace);
  missingCoordinate.expertOutputPreview[0].pop();
  assert.throws(() => sumCoordinate(missingCoordinate, 7, 0.22), /coordinate values must be valid/);
  const invalidRecorded = structuredClone(trace);
  invalidRecorded.combinedOutputPreview[0] = Infinity;
  assert.throws(() => sumCoordinate(invalidRecorded, 0, 0.22), /must be finite/);
});
