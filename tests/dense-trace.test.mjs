import test from 'node:test';
import assert from 'node:assert/strict';

const examples = await Promise.all(
	Array.from({ length: 5 }, async (_, index) => {
		const fixture = await import(`../dense/src/constants/examples/ex${index}.js`);
		return fixture[`ex${index}`];
	})
);

const LAYERS = 12;
const HEADS = 12;
const HEAD_SIZE = 64;
const VOCAB_SIZE = 50257;
const ATTENTION_STAGES = ['attn', 'attn_scaled', 'attn_masked', 'attn_softmax', 'attn_dropout'];
// The saved attention softmax is float32; this reconstruction uses JavaScript float64.
const FLOAT32_TOLERANCE = 2 ** -22;
// These presets were recorded with the upstream default sampling controls.
const TEMPERATURE = 0.8;
const TOP_K = 5;

function near(actual, expected, tolerance, label) {
	assert.ok(Number.isFinite(actual), `${label}: non-finite value`);
	assert.ok(
		Math.abs(actual - expected) <= tolerance,
		`${label}: ${actual} differs from ${expected}`
	);
}

for (const [index, example] of examples.entries()) {
	test(`dense preset ${index}: all attention heads preserve scaling, causal masking and softmax`, () => {
		const length = example.tokens.length;
		assert.ok(length > 0);
		assert.equal(example.tokens.join(''), example.prompt);
		assert.equal(example.tokenIds.length, length);
		assert.ok(example.tokenIds.every((id) => Number.isInteger(id) && id >= 0 && id < VOCAB_SIZE));
		assert.equal(Object.keys(example.outputs).length, LAYERS * HEADS * ATTENTION_STAGES.length);

		for (let layer = 0; layer < LAYERS; layer++) {
			for (let head = 0; head < HEADS; head++) {
				const label = `preset ${index}, layer ${layer}, head ${head}`;
				const matrices = ATTENTION_STAGES.map((stage) => {
					const matrix = example.outputs[`block_${layer}_attn_head_${head}_${stage}`]?.data;
					assert.ok(Array.isArray(matrix), `${label}, ${stage}: missing matrix`);
					assert.equal(matrix.length, length, `${label}, ${stage}: row count`);
					for (const row of matrix) {
						assert.ok(Array.isArray(row));
						assert.equal(row.length, length, `${label}, ${stage}: column count`);
					}
					return matrix;
				});
				const [raw, scaled, masked, softmax, dropout] = matrices;

				for (let row = 0; row < length; row++) {
					const maximum = Math.max(...scaled[row].slice(0, row + 1));
					const exponents = scaled[row].map((value, column) =>
						column <= row ? Math.exp(value - maximum) : 0
					);
					const denominator = exponents.reduce((sum, value) => sum + value, 0);
					for (let column = 0; column < length; column++) {
						const cell = `${label}, [${row},${column}]`;
						assert.ok(Number.isFinite(raw[row][column]), `${cell}: raw score`);
						assert.equal(scaled[row][column], raw[row][column] / Math.sqrt(HEAD_SIZE));
						if (column > row) {
							// JSON.stringify serializes the recorded -Infinity causal mask as null.
							assert.equal(masked[row][column], null, `${cell}: future mask`);
							assert.equal(softmax[row][column], 0, `${cell}: future attention`);
						} else {
							assert.equal(masked[row][column], scaled[row][column], `${cell}: allowed score`);
						}
						assert.ok(softmax[row][column] >= 0 && softmax[row][column] <= 1);
						near(softmax[row][column], exponents[column] / denominator, FLOAT32_TOLERANCE, cell);
						assert.equal(dropout[row][column], softmax[row][column], `${cell}: eval dropout`);
					}
					near(
						softmax[row].reduce((sum, value) => sum + value, 0),
						1,
						FLOAT32_TOLERANCE,
						`${label}, row ${row}: probability mass`
					);
				}
			}
		}
	});

	test(`dense preset ${index}: full vocabulary logits reproduce the recorded top-k prediction`, () => {
		assert.equal(example.logits.length, VOCAB_SIZE);
		assert.ok(example.logits.every(Number.isFinite));
		assert.equal(example.probabilities.length, 50);
		const ranked = example.logits
			.map((logit, tokenId) => ({ logit, tokenId }))
			.sort((a, b) => b.logit - a.logit);
		const maximum = ranked[0].logit / TEMPERATURE;
		const exponents = ranked
			.slice(0, TOP_K)
			.map(({ logit }) => Math.exp(logit / TEMPERATURE - maximum));
		const denominator = exponents.reduce((sum, value) => sum + value, 0);
		for (const [rank, candidate] of example.probabilities.entries()) {
			assert.equal(candidate.rank, rank);
			assert.equal(candidate.tokenId, ranked[rank].tokenId);
			assert.equal(candidate.logit, example.logits[candidate.tokenId]);
			assert.equal(candidate.scaledLogit, candidate.logit / TEMPERATURE);
			assert.equal(candidate.topKLogit, rank < TOP_K ? candidate.scaledLogit : null);
			near(candidate.expLogit, exponents[rank] ?? 0, 1e-12, `rank ${rank}: exponent`);
			near(
				candidate.probability,
				rank < TOP_K ? exponents[rank] / denominator : 0,
				1e-12,
				`rank ${rank}: sampled-distribution probability`
			);
		}
		near(
			example.probabilities.reduce((sum, candidate) => sum + candidate.probability, 0),
			1,
			1e-12,
			'next-token probability mass'
		);
		assert.ok(example.sampled.rank >= 0 && example.sampled.rank < TOP_K);
		assert.deepEqual(example.sampled, example.probabilities[example.sampled.rank]);
	});
}
