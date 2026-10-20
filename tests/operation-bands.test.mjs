import { test } from 'node:test';
import assert from 'node:assert/strict';
import { operationGraph } from '../src/components/common/operation-graphs.js';

const node = (g, id) => g.nodes.find((n) => n.id === id);
const linked = (g, from, to) => g.edges.some((e) => e.from === from && e.to === to);
test('QKV bands branch from one input and preserve unequal query/KV head counts', () => {
	const g = operationGraph('qkv', { hiddenSize: 576, heads: 9, kvHeads: 3, headSize: 64 });
	for (const id of ['query', 'key', 'value']) assert.ok(linked(g, 'input', id));
	assert.equal(node(g, 'queryOut').dim, '9 × 64D');
	assert.equal(node(g, 'keyOut').dim, '3 × 64D');
	assert.equal(node(g, 'valueOut').dim, '3 × 64D');
	assert.ok(!linked(g, 'query', 'key'));
});
test('cross-attention keeps source length and routes V only to the weighted sum', () => {
	const g = operationGraph('attention', { headSize: 64, sequenceLength: 11, scale: 'unscaled' });
	assert.equal(node(g, 'k').dim, '11 × 64');
	assert.equal(node(g, 'scores').dim, '11 scores');
	assert.ok(linked(g, 'q', 'scores') && linked(g, 'k', 'scores'));
	assert.ok(linked(g, 'v', 'context') && !linked(g, 'v', 'scores'));
	assert.ok(linked(g, 'scores', 'softmax') && linked(g, 'softmax', 'context'));
	assert.equal(node(g, 'softmax').dim, 'unscaled');
});
test('Mamba gate and time-step bypass convolution and meet the correct branch', () => {
	const g = operationGraph('mamba');
	for (const [a, b] of [
		['projection', 'convolution'],
		['convolution', 'scan'],
		['projection', 'z'],
		['z', 'gate'],
		['projection', 'dt'],
		['dt', 'scan'],
		['scan', 'gate'],
		['gate', 'output']
	])
		assert.ok(linked(g, a, b), `${a} → ${b}`);
	assert.ok(!linked(g, 'z', 'convolution') && !linked(g, 'dt', 'convolution'));
	assert.equal(node(g, 'projection').dim, '2048 → 8512');
	assert.equal(node(g, 'output').dim, '2048D');
});
test('rotary branches leave the value path unchanged and noninteractive', () => {
	const g = operationGraph('rope', { headSize: 128 });
	assert.ok(linked(g, 'q', 'query') && linked(g, 'k', 'key'));
	assert.ok(linked(g, 'v', 'vo'));
	assert.equal(g.edges.find((e) => e.from === 'v').action, '');
	assert.equal(node(g, 'qo').label, 'Q′');
});
test('vision connector preserves pixel shuffle dimensions and all graph endpoints resolve', () => {
	const g = operationGraph('connector');
	assert.equal(node(g, 'pack').dim, '16 × 12288');
	assert.equal(node(g, 'out').dim, '16 × 576');
	for (const kind of [
		'qkv',
		'attention',
		'mamba',
		'rope',
		'residual',
		'connector',
		'embedding',
		'language-embedding',
		'norm',
		'classifier'
	]) {
		const graph = operationGraph(kind);
		const ids = new Set(graph.nodes.map((n) => n.id));
		assert.equal(ids.size, graph.nodes.length);
		for (const edge of graph.edges) assert.ok(ids.has(edge.from) && ids.has(edge.to), kind);
	}
});

test('band widths carry operation dimensions through compression and expansion', () => {
	const qkv = operationGraph('qkv', { hiddenSize: 576, heads: 9, kvHeads: 3, headSize: 64 });
	assert.deepEqual(
		qkv.edges.map((e) => e.dimension),
		[576, 576, 576, 192, 576, 192]
	);
	const patches = operationGraph('embedding', { hiddenSize: 768, outputSize: 192 });
	assert.deepEqual(
		patches.edges.map((e) => e.dimension),
		[768, 192, 192]
	);
	const connector = operationGraph('connector');
	assert.deepEqual(
		connector.edges.map((e) => e.dimension),
		[768, 12288, 576]
	);
	const mamba = operationGraph('mamba');
	assert.equal(mamba.edges.find((e) => e.to === 'dt').dimension, 64);
	assert.equal(mamba.edges.find((e) => e.to === 'z').dimension, 4096);
	assert.equal(mamba.edges.find((e) => e.to === 'convolution').dimension, 4352);
});
