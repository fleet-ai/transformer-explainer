import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const compiled = ts.transpileModule(
	readFileSync(new URL('../src/lib/architectures/types.ts', import.meta.url), 'utf8'),
	{ compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }
).outputText;
const exports = {};
runInNewContext(compiled, { exports });
const { tensorRowForPosition, tensorRows } = exports;
const recording = (id) =>
	JSON.parse(readFileSync(new URL(`../static/architectures/${id}.json`, import.meta.url), 'utf8'));

test('a repeated token keeps its actual occurrence in the recorded sequence', () => {
	const example = recording('hybrid').examples.find(
		(item) => item.tokens.filter((token) => token === 'a').length > 1
	);
	assert.ok(example);
	const tensor = example.layers[0].operations.find((op) => op.id === 'embedding').tensors[0];
	const positions = example.tokens
		.map((token, index) => (token === 'a' ? index : -1))
		.filter((index) => index >= 0);
	for (const position of positions)
		assert.equal(tensorRowForPosition(tensor, position, 'a'), position);
});

test('a sparse vision attention slice resolves original patch labels, not stored row numbers', () => {
	const tensor = recording('vision-language').examples[0].layers[0].operations.find(
		(op) => op.id === 'vision-attention'
	).tensors[0];
	const saved = tensorRowForPosition(tensor, 31, 'patch 0,31');
	assert.equal(saved, tensor.rowLabels.indexOf('patch 0,31'));
	assert.notEqual(saved, 31);
	assert.equal(tensorRowForPosition(tensor, 100, 'patch 3,4'), -1);
});

test('source selection does not silently select a decoder row with the same numeric index', () => {
	const example = recording('encoder-decoder').examples[0];
	const tensor = example.layers[0].operations.find((op) => op.id === 'decoder-self').tensors[0];
	assert.equal(tensorRowForPosition(tensor, 1, example.tokens[1]), -1);
});

test('one-dimensional labelled class probabilities keep each class with its recorded value', () => {
	const tensor = recording('vit').examples[0].layers[0].operations.find((op) => op.id === 'output')
		.tensors[0];
	const rows = tensorRows(tensor);
	assert.equal(rows.length, tensor.rowLabels.length);
	for (let index = 0; index < rows.length; index++) {
		assert.equal(rows[index].length, 1);
		assert.equal(rows[index][0], tensor.values[index]);
	}
});

test('a coordinate vector without row labels stays a single vector', () => {
	const rows = tensorRows({
		name: 'CLS coordinates',
		shape: [192],
		values: [0.1, -0.2, 0.3],
		slice: 'Coordinates 0–2'
	});
	assert.equal(rows.length, 1);
	assert.equal(rows[0].length, 3);
});
