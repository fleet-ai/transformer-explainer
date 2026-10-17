import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const near = (actual, expected, label, absolute = 1e-5, relative = 2e-6) => {
  assert.ok(Number.isFinite(actual) && Number.isFinite(expected)
    && Math.abs(actual - expected) <= absolute + relative * Math.abs(expected), label);
};
const silu = x => x >= 0 ? x / (1 + Math.exp(-x)) : x * Math.exp(x) / (1 + Math.exp(x));
const vector = (values, label, length = 4) => {
  assert.equal(values.length, length, `${label}: coordinate count`);
  assert.ok(values.every(Number.isFinite), `${label}: non-finite coordinate`);
};

/** Validate actual captured arithmetic, including signed omitted dot terms. */
export function validateExpertOperations(sidecar, main) {
  assert.equal(sidecar.schemaVersion, 1);
  assert.equal(sidecar.kind, 'expert-operations');
  assert.equal(sidecar.model.id, main.model.id);
  assert.equal(sidecar.model.revision, main.model.revision);
  assert.match(sidecar.capture.scriptSha256, /^[a-f0-9]{64}$/);
  assert.match(sidecar.capture.mainBundleSha256, /^[a-f0-9]{64}$/);
  assert.equal(sidecar.capture.matchesMainVocabularyLogits, true);
  assert.equal(sidecar.capture.omittedDotProductDtype, 'float64');
  const example = main.examples.find(item => item.id === sidecar.example.id);
  assert.ok(example, 'Example is present in main bundle');
  assert.equal(sidecar.example.prompt, example.prompt);
  assert.deepEqual(sidecar.example.tokens, example.tokens);
  assert.equal(sidecar.layers.length, main.model.numLayers);
  let expertEvaluations = 0;
  let projections = 0;
  let signedNegativeOmittedTerms = 0;
  let maxProjectionReconstructionError = 0;
  let maxWeightedSumPreviewError = 0;

  function validateProjection(projection, input, inputDimensions) {
    projections++;
    assert.equal(projection.inputDimensions, inputDimensions, 'Projection input dimension');
    assert.equal(projection.omittedDimensions, inputDimensions - 4, 'Omitted coordinate count');
    assert.equal(projection.weights.length, 4);
    projection.weights.forEach(row => vector(row, 'Weight row'));
    vector(projection.omittedContribution, 'Signed omitted contribution');
    vector(projection.output, 'Actual projection output');
    vector(projection.reconstructionTolerance, 'Projection tolerance');
    for (let row = 0; row < 4; row++) {
      const omitted = projection.omittedContribution[row];
      if (omitted < 0) signedNegativeOmittedTerms++;
      const reconstructed = projection.weights[row].reduce((sum, weight, column) => sum + weight * input[column], 0) + omitted;
      const error = Math.abs(reconstructed - projection.output[row]);
      const tolerance = projection.reconstructionTolerance[row];
      assert.ok(tolerance > 0 && error <= tolerance, 'Visible dot product plus signed omitted term must reconstruct actual output');
      maxProjectionReconstructionError = Math.max(maxProjectionReconstructionError, error);
    }
  }

  for (const [layerIndex, layer] of sidecar.layers.entries()) {
    assert.equal(layer.index, layerIndex);
    assert.equal(layer.tokens.length, example.tokens.length);
    for (const [tokenIndex, token] of layer.tokens.entries()) {
      const original = example.layers[layerIndex].tokens[tokenIndex];
      assert.deepEqual(token.selectedExperts, original.selectedExperts);
      assert.deepEqual(token.routingWeights, original.routingWeights);
      assert.deepEqual(token.combinedOutput, original.combinedOutputPreview.slice(0, 4));
      vector(token.inputPreview, 'Actual MoE input');
      vector(token.combinedOutput, 'Actual combined output');
      assert.equal(token.experts.length, main.model.topK);
      for (const [rank, expert] of token.experts.entries()) {
        expertEvaluations++;
        assert.equal(expert.id, token.selectedExperts[rank], 'Expert ordering follows routing rank');
        validateProjection(expert.gate, token.inputPreview, main.model.hiddenSize);
        validateProjection(expert.up, token.inputPreview, main.model.hiddenSize);
        vector(expert.siluGate, 'Actual SiLU output');
        vector(expert.product, 'Actual elementwise product');
        vector(expert.weightedOutput, 'Actual weighted expert output');
        validateProjection(expert.down, expert.product, main.model.intermediateSize);
        assert.deepEqual(expert.down.output, original.expertOutputPreview[rank].slice(0, 4));
        for (let coordinate = 0; coordinate < 4; coordinate++) {
          near(expert.siluGate[coordinate], silu(expert.gate.output[coordinate]), 'SiLU gate', 2e-6);
          near(expert.product[coordinate], expert.siluGate[coordinate] * expert.up.output[coordinate], 'SiLU(gate) × up product');
          near(expert.weightedOutput[coordinate], expert.down.output[coordinate] * token.routingWeights[rank], 'Routing weight × expert output');
        }
      }
      for (let coordinate = 0; coordinate < 4; coordinate++) {
        const sum = token.experts.reduce((value, expert) => value + expert.weightedOutput[coordinate], 0);
        near(token.combinedOutput[coordinate], sum, 'Sum of actual weighted expert outputs');
        maxWeightedSumPreviewError = Math.max(maxWeightedSumPreviewError, Math.abs(token.combinedOutput[coordinate] - sum));
      }
    }
  }
  return { expertEvaluations, projections, signedNegativeOmittedTerms, maxProjectionReconstructionError, maxWeightedSumPreviewError };
}

export function verifyExpertOperationsProvenance(sidecar, mainBytes, sourceBytes) {
  assert.equal(sidecar.capture.mainBundleSha256, createHash('sha256').update(mainBytes).digest('hex'), 'Main bundle hash');
  assert.equal(sidecar.capture.scriptSha256, createHash('sha256').update(sourceBytes).digest('hex'), 'Expert capture source hash');
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const root = process.argv[2] || 'static/traces';
  const mainBytes = readFileSync(`${root}/granite-moe.json`);
  const main = JSON.parse(mainBytes);
  const sourceBytes = readFileSync('scripts/capture_expert_operations.py');
  for (const example of main.examples) {
    const sidecar = JSON.parse(readFileSync(`${root}/expert-operations-${example.id}.json`, 'utf8'));
    verifyExpertOperationsProvenance(sidecar, mainBytes, sourceBytes);
    console.log(example.id, JSON.stringify(validateExpertOperations(sidecar, main)));
  }
}
