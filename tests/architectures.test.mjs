import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const ids = ['mla', 'encoder-decoder', 'hybrid', 'vision-language', 'vit'];
const traces = new Map();
for (const id of ids) {
	const trace = JSON.parse(
		await readFile(new URL(`../static/architectures/${id}.json`, import.meta.url), 'utf8')
	);
	traces.set(id, trace);
	test(`${id}: recorded examples have finite tensors and explicit checkpoint provenance`, async () => {
		assert.equal(trace.schemaVersion, 1);
		assert.equal(trace.architecture, id);
		assert.equal(trace.provenance.trained, true);
		assert.match(trace.provenance.revision, /^[a-f0-9]{40}$/);
		assert.ok(trace.provenance.scope.length > 30);
		assert.ok(trace.examples.length > 0);
		if (trace.provenance.captureScriptSha256) {
			const script = await readFile(
				new URL(`../${trace.provenance.captureScript}`, import.meta.url)
			);
			assert.equal(
				createHash('sha256').update(script).digest('hex'),
				trace.provenance.captureScriptSha256
			);
		}
		for (const check of [
			...(trace.verification ?? []),
			...trace.examples.flatMap((example) => example.verification ?? [])
		]) {
			assert.ok(Number.isFinite(check.maxAbsoluteError));
			assert.ok(check.maxAbsoluteError <= check.tolerance, `${id}: ${check.name}`);
		}
		for (const example of trace.examples) {
			assert.ok(example.layers.length > 0);
			for (const layer of example.layers) {
				assert.ok(layer.operations.length > 0);
				assert.equal(new Set(layer.operations.map((op) => op.id)).size, layer.operations.length);
				for (const op of layer.operations) {
					assert.ok(op.formula?.length > 0, `${id}: ${op.id} formula`);
					for (const tensor of op.tensors) {
						assert.ok(tensor.shape.every((n) => Number.isInteger(n) && n > 0));
						assert.ok(tensor.values.length > 0);
						const flat = tensor.values.flat(Infinity);
						assert.ok(
							flat.every(Number.isFinite),
							`${id}: ${tensor.name} contains invalid numbers`
						);
						if (Array.isArray(tensor.values[0])) {
							assert.ok(tensor.values.every((row) => row.length === tensor.values[0].length));
							if (tensor.rowLabels) assert.equal(tensor.rowLabels.length, tensor.values.length);
							if (tensor.columnLabels)
								assert.equal(tensor.columnLabels.length, tensor.values[0].length);
						}
						if (tensor.role === 'attention' || tensor.role === 'probabilities') {
							assert.ok(
								flat.every((n) => n >= 0 && n <= 1.00001),
								`${id}: probability range`
							);
						}
						// Only full attention rows have unit mass; saved crops deliberately do not.
						if (
							tensor.role === 'attention' &&
							Array.isArray(tensor.values[0]) &&
							tensor.values[0].length === tensor.shape.at(-1)
						) {
							for (const row of tensor.values)
								assert.ok(
									Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 2e-5,
									`${id}: attention row mass`
								);
						}
					}
				}
			}
		}
	});
}

test('T5 cross-attention reproduces the recorded context using encoder values', () => {
	for (const example of traces.get('encoder-decoder').examples) {
		for (const layer of example.layers) {
			const op = layer.operations.find((op) => op.id === 'cross-attention');
			const [attention, values, context] = op.tensors.map((t) => t.values);
			for (let row = 0; row < attention.length; row++)
				for (let col = 0; col < 4; col++) {
					const computed = attention[row].reduce(
						(sum, weight, i) => sum + weight * values[i][col],
						0
					);
					assert.ok(Math.abs(computed - context[row][col]) < 2e-5);
				}
			const causal = layer.operations.find((op) => op.id === 'decoder-self').tensors[0].values;
			for (let row = 0; row < causal.length; row++)
				for (let col = row + 1; col < causal[row].length; col++) assert.equal(causal[row][col], 0);
			const bidirectional = layer.operations.find((op) => op.id === 'encoder-self').tensors[0]
				.values;
			assert.ok(bidirectional[0].slice(1).some((p) => p > 0));
		}
	}
});

test('partial pretrained captures never claim a full-model prediction', () => {
	for (const id of ['mla', 'hybrid']) {
		const trace = traces.get(id);
		assert.equal(trace.provenance.partial, true);
		for (const example of trace.examples) {
			assert.ok(!example.output?.text);
			assert.ok(!example.output?.predictions?.length);
		}
	}
});

test('MLA attention applies the recorded value vectors for every prompt position', () => {
	for (const example of traces.get('mla').examples) {
		const ops = example.layers[0].operations;
		const values = ops
			.find((op) => op.id === 'expand')
			.tensors.find((t) => t.name === 'V head 1').values;
		const attentionOp = ops.find((op) => op.id === 'attention');
		const attention = attentionOp.tensors.find((t) => t.head === 0).values;
		const context = attentionOp.tensors.find((t) => t.name === 'Head 1 attended values').values;
		for (let row = 0; row < attention.length; row++)
			for (let col = 0; col < context[row].length; col++) {
				const computed = attention[row].reduce(
					(sum, weight, i) => sum + weight * values[i][col],
					0
				);
				assert.ok(Math.abs(computed - context[row][col]) < 2e-5);
			}
		for (const tensor of attentionOp.tensors.filter((t) => t.role === 'attention')) {
			for (let row = 0; row < tensor.values.length; row++)
				for (let col = row + 1; col < tensor.values[row].length; col++)
					assert.equal(tensor.values[row][col], 0);
		}
	}
	const c = traces.get('mla').config;
	assert.equal(c.cacheCoordinatesPerToken, c.kvLatentDim + c.rotaryDim);
	assert.equal(
		c.expandedKVCoordinatesPerToken,
		c.attentionHeads * (c.contentKeyDim + c.rotaryDim + c.valueDim)
	);
	assert.ok(c.cacheCoordinatesPerToken < c.expandedKVCoordinatesPerToken);
});

test('Mamba state carries the previous token through the recorded decay and injection', () => {
	for (const example of traces.get('hybrid').examples)
		for (const layer of example.layers) {
			const op = layer.operations.find((op) => op.id === 'selective-state');
			const find = (name) => op.tensors.find((t) => t.name === name).values;
			const prev = find('Previous state S_(t−1)');
			const decay = find('Decay exp(Δ A)');
			const injection = find('Input injection Δ B x');
			const state = find('Updated state S_t');
			for (let t = 0; t < state.length; t++)
				for (let c = 0; c < state[t].length; c++) {
					assert.equal(prev[t][c], t === 0 ? 0 : state[t - 1][c]);
					const expected = decay[t][c] * prev[t][c] + injection[t][c];
					assert.ok(Math.abs(state[t][c] - expected) < 1e-6);
				}
		}
});
