import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const near = (actual, expected, tolerance, label) => assert.ok(
  Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
  `${label}: ${actual} != ${expected} (tolerance ${tolerance})`
);
const softmax = values => {
  const max = Math.max(...values);
  const exp = values.map(value => Math.exp(value - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(value => value / sum);
};
function array(value, length, label) {
  assert.ok(Array.isArray(value), `${label} missing or not an array`);
  assert.equal(value.length, length, `${label} length`);
}
function vector(value, length, label) {
  array(value, length, label);
  assert.ok(value.every(Number.isFinite), `${label} contains non-finite values`);
}
function previews(values, norms, tokenCount, label) {
  array(values, tokenCount, `${label} previews`);
  vector(norms, tokenCount, `${label} norms`);
  values.forEach((value, token) => {
    vector(value, 8, `${label} coordinates`);
    assert.ok(norms[token] >= 0, `${label} norm is negative`);
    assert.ok(Math.hypot(...value) <= norms[token] + 1e-5, `${label} preview exceeds full-vector norm`);
  });
}

/** Validate the actual recorded arithmetic, not only JSON shape. */
export function validateTraces(data) {
  assert.equal(data.schemaVersion, 1);
  const { model, capture, examples } = data;
  assert.match(model.revision, /^[a-f0-9]{40}$/);
  assert.ok(model.id && model.url && model.license, 'Model provenance missing');
  assert.ok(Number.isFinite(Date.parse(capture.timestamp)), 'Capture timestamp missing');
  assert.ok(capture.device && capture.dtype && capture.packages, 'Capture environment missing');
  assert.ok(model.topK > 0 && model.topK <= model.numExperts);
  assert.ok(Number.isInteger(model.attentionHeads) && model.attentionHeads > 0, 'Attention head count missing');
  assert.ok(examples.length >= 3, 'Expected language, code and math captures');
  assert.equal(new Set(examples.map(e => e.id)).size, examples.length);
  let tokenLayerPairs = 0;
  let attentionRows = 0;
  let maxReconstructionError = 0;
  for (const example of examples) {
    assert.ok(example.prompt.length > 0 && example.tokens.length > 0);
    assert.equal(example.layers.length, model.numLayers, 'Missing captured layer');
    example.tokens.forEach((token, index) => {
      assert.equal(token.index, index, 'Token alignment');
      assert.ok(Number.isInteger(token.id) && typeof token.text === 'string');
    });
    const tokenCount = example.tokens.length;
    previews(example.embeddingPreview, example.embeddingNorms, tokenCount, 'Embedding');
    previews(example.finalHiddenPreview, example.finalHiddenNorms, tokenCount, 'Final hidden');
    array(example.attention, model.numLayers, 'Attention layers');
    for (const heads of example.attention) {
      array(heads, model.attentionHeads, 'Attention heads');
      for (const matrix of heads) {
        array(matrix, tokenCount, 'Attention queries');
        matrix.forEach((row, query) => {
          attentionRows++;
          vector(row, tokenCount, 'Attention keys');
          assert.ok(row.every(value => value >= 0 && value <= 1), 'Attention probabilities out of range');
          near(row.reduce((sum, value) => sum + value, 0), 1, 2e-7, 'Attention row probability mass');
          assert.ok(row.slice(query + 1).every(value => value === 0), 'Future-token attention must be masked');
        });
      }
    }
    for (const [layerIndex, layer] of example.layers.entries()) {
      assert.equal(layer.index, layerIndex, 'Layer alignment');
      assert.equal(layer.tokens.length, example.tokens.length, 'Token-layer alignment');
      previews(layer.attentionInputPreview, layer.attentionInputNorms, tokenCount, 'Attention input');
      previews(layer.attentionOutputPreview, layer.attentionOutputNorms, tokenCount, 'Attention output');
      const counts = Array(model.numExperts).fill(0);
      for (const row of layer.tokens) {
        tokenLayerPairs++;
        vector(row.routerLogits, model.numExperts, 'Router logits');
        vector(row.routerProbabilities, model.numExperts, 'Router probabilities');
        const expectedProbabilities = softmax(row.routerLogits);
        row.routerProbabilities.forEach((p, i) => near(p, expectedProbabilities[i], 1e-5, 'Router softmax'));
        near(row.routerProbabilities.reduce((a, b) => a + b, 0), 1, 1e-5, 'Probability mass');
        assert.equal(row.selectedExperts.length, model.topK);
        assert.equal(new Set(row.selectedExperts).size, model.topK, 'Duplicate selected expert');
        const sortedScores = [...row.routerLogits].sort((a, b) => b - a);
        const selectedLogits = row.selectedExperts.map((expert, rank) => {
          assert.ok(Number.isInteger(expert) && expert >= 0 && expert < model.numExperts);
          counts[expert]++;
          near(row.routerLogits[expert], sortedScores[rank], 1e-5, 'Top-k selection');
          return row.routerLogits[expert];
        });
        vector(row.routingWeights, model.topK, 'Routing weights');
        const expectedWeights = softmax(selectedLogits);
        row.routingWeights.forEach((weight, i) => near(weight, expectedWeights[i], 1e-5, 'Selected weight'));
        near(row.routingWeights.reduce((a, b) => a + b, 0), 1, 1e-5, 'Selected weight sum');
        vector(row.expertOutputNorms, model.topK, 'Expert norms');
        vector(row.weightedOutputNorms, model.topK, 'Weighted norms');
        row.expertOutputNorms.forEach((norm, i) => {
          assert.ok(norm >= 0);
          near(row.weightedOutputNorms[i], norm * row.routingWeights[i], Math.max(1e-5, norm * 1e-5), 'Weighted norm');
        });
        assert.equal(row.expertOutputPreview.length, model.topK);
        row.expertOutputPreview.forEach(v => vector(v, 8, 'Expert coordinates'));
        vector(row.combinedOutputPreview, 8, 'Combined coordinates');
        row.combinedOutputPreview.forEach((coordinate, dimension) => {
          const sum = row.expertOutputPreview.reduce((total, expert, i) => total + row.routingWeights[i] * expert[dimension], 0);
          near(coordinate, sum, Math.max(1e-5, Math.abs(coordinate) * 1e-5), 'Weighted sum coordinate');
        });
        assert.ok(Number.isFinite(row.inputNorm) && row.inputNorm >= 0);
        assert.ok(Number.isFinite(row.outputNorm) && row.outputNorm >= 0);
        assert.ok(Number.isFinite(row.outputMaxAbs) && row.outputMaxAbs >= 0, 'Output scale missing');
        assert.ok(Number.isFinite(row.reconstructionToleranceRatio) && row.reconstructionToleranceRatio >= 0 && row.reconstructionToleranceRatio <= 1, 'Full-vector relative reconstruction failed');
        assert.ok(row.reconstructionMaxAbsError >= 0 && row.reconstructionMaxAbsError <= 1e-5 + 2e-6 * row.outputMaxAbs, 'Full-vector reconstruction failed');
        maxReconstructionError = Math.max(maxReconstructionError, row.reconstructionMaxAbsError);
      }
      if (layer.expertTokenCounts) assert.deepEqual(layer.expertTokenCounts, counts, 'Expert load accounting');
      assert.equal(counts.reduce((a, b) => a + b, 0), example.tokens.length * model.topK);
    }
    assert.ok(example.nextTokenCandidates.length > 0);
    let previous = 1;
    let totalProbability = 0;
    for (const candidate of example.nextTokenCandidates) {
      assert.ok(Number.isInteger(candidate.id) && typeof candidate.text === 'string');
      assert.ok(candidate.probability >= 0 && candidate.probability <= previous, 'Candidate ranking');
      previous = candidate.probability;
      totalProbability += candidate.probability;
    }
    assert.ok(totalProbability > 0 && totalProbability <= 1.00001, 'Candidate probability mass');
  }
  return { examples: examples.length, tokenLayerPairs, attentionRows, expertEvaluations: tokenLayerPairs * model.topK, maxReconstructionError };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const path = process.argv[2] || 'static/traces/granite-moe.json';
  console.log(JSON.stringify(validateTraces(JSON.parse(readFileSync(path, 'utf8'))), null, 2));
}
