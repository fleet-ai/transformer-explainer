import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, statSync } from 'node:fs';
import { validateExpertOperations, verifyExpertOperationsProvenance } from '../scripts/validate-expert-operations.mjs';

const root = new URL('../static/traces/', import.meta.url);
const mainBytes = readFileSync(new URL('granite-moe.json', root));
const main = JSON.parse(mainBytes);
const sidecars = main.examples.map(example => JSON.parse(readFileSync(new URL(`expert-operations-${example.id}.json`, root), 'utf8')));
const fresh = () => structuredClone(sidecars[0]);

test('real expert sidecars reconstruct all gate/up/down projections, SiLU products and weighted sums', () => {
  let totalExperts = 0;
  let totalProjections = 0;
  for (const data of sidecars) {
    const report = validateExpertOperations(data, main);
    totalExperts += report.expertEvaluations;
    totalProjections += report.projections;
    assert.ok(report.signedNegativeOmittedTerms > 0, 'Real omitted terms retain their signs');
    assert.ok(statSync(new URL(`expert-operations-${data.example.id}.json`, root)).size < 25_000_000);
  }
  assert.equal(totalExperts, 5568);
  assert.equal(totalProjections, 16704);
});

test('rejects an omitted dot contribution replaced by an unrelated value', () => {
  const data = fresh();
  data.layers[0].tokens[0].experts[0].gate.omittedContribution[0] += 100;
  assert.throws(() => validateExpertOperations(data, main));
});

test('rejects an incorrect visible weight in the real projection', () => {
  const data = fresh();
  data.layers[0].tokens[0].experts[0].up.weights[0][0] += 100;
  assert.throws(() => validateExpertOperations(data, main));
});

test('rejects down projection fed with gate values instead of the actual SwiGLU product', () => {
  const data = fresh();
  const expert = data.layers[0].tokens[0].experts[0];
  expert.product = [...expert.gate.output];
  assert.throws(() => validateExpertOperations(data, main));
});

test('rejects an incorrect SiLU activation, expert ID, and routing weight', () => {
  for (const mutate of [
    data => { data.layers[0].tokens[0].experts[0].siluGate[0] += 1; },
    data => { data.layers[0].tokens[0].experts[0].id = -1; },
    data => { data.layers[0].tokens[0].routingWeights[0] += 0.1; }
  ]) {
    const data = fresh();
    mutate(data);
    assert.throws(() => validateExpertOperations(data, main));
  }
});

test('expert-operation provenance points to unchanged main bundle and exact downloadable source', () => {
  const source = readFileSync(new URL('../scripts/capture_expert_operations.py', import.meta.url));
  const downloaded = readFileSync(new URL('capture_expert_operations.py', root));
  assert.deepEqual(source, downloaded);
  for (const data of sidecars) verifyExpertOperationsProvenance(data, mainBytes, source);
});
