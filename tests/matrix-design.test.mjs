import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { compile } from 'svelte/compiler';
import { render } from 'svelte/server';
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
const directory = await mkdtemp(new URL('./.matrix-render-', import.meta.url));
after(() => rm(directory, { recursive: true, force: true, maxRetries: 3 }));
async function component(name, folder = 'common') {
	const source = await readFile(
		new URL(`../src/components/${folder}/${name}.svelte`, import.meta.url),
		'utf8'
	);
	const file = `${directory}/${name}.mjs`;
	await writeFile(
		file,
		compile(source, { filename: `${name}.svelte`, generate: 'server' }).js.code
	);
	return (await import(pathToFileURL(file))).default;
}
const Matrix = await component('MatrixProduct');
const Flow = await component('FeedForwardFlow');
test('matrix view preserves rectangular dimensions and recorded preview', () => {
	const { body } = render(Matrix, {
		props: { inputSize: 512, outputSize: 1024, inputValues: [0.125], outputValues: [-0.25] }
	});
	assert.match(body, /1 × 512/);
	assert.match(body, /512 × 1024/);
	assert.match(body, /1 × 1024/);
	assert.match(body, /recorded: 0.125000/);
	assert.match(body, /recorded: -0.250000/);
});
test('elementwise products and activations preserve intermediate shape', () => {
	const product = render(Matrix, {
		props: { inputSize: 8192, outputSize: 8192, elementwise: true }
	}).body;
	assert.match(product, /⊙/);
	assert.doesNotMatch(product, /8192 × 8192/);
	const activation = render(Matrix, {
		props: { inputSize: 2048, outputSize: 2048, activation: 'ReLU' }
	}).body;
	assert.match(activation, /through ReLU/);
	assert.match(activation, /elementwise/);
});
test('biased projections explicitly show the bias in equation', () => {
	const { body } = render(Matrix, { props: { inputSize: 192, outputSize: 768, bias: true } });
	assert.match(body, /plus bias equals/);
	assert.match(body, /\+ b<\/small>/);
	assert.match(body, /bias \(768\)/);
});
test('gated flow exposes each branch and independent activation/product controls', () => {
	const { body } = render(Flow, { props: { selected: 'gate' } });
	for (const label of [
		'up band',
		'gate band',
		'product band',
		'down band',
		'up operation',
		'gate operation',
		'SiLU operation',
		'product operation',
		'down operation'
	])
		assert.ok(body.includes(`Inspect ${label}`), label);
	assert.match(body, /Inspect gate operation" aria-pressed="true"/);
	assert.match(body, /Inspect product operation" aria-pressed="false"/);
});
test('Dense-style flow omits the gate and exposes GELU', () => {
	const { body } = render(Flow, {
		props: { gated: false, activation: 'GELU', hiddenSize: 768, intermediateSize: 3072 }
	});
	assert.doesNotMatch(body, /Inspect gate/);
	assert.match(body, /Inspect GELU operation/);
	assert.match(body, /3072/);
});

const LatentFlow = await component('LatentProjectionFlow', 'mla');
test('MLA flow renders one query and KV projection with separate query and shared-key RoPE controls', () => {
	const { body } = render(LatentFlow, { props: { selected: 'compression' } });
	assert.equal((body.match(/data-node="query"/g) ?? []).length, 1);
	assert.equal((body.match(/data-node="compression"/g) ?? []).length, 1);
	assert.equal((body.match(/RoPE · Rotate Q and K/g) ?? []).length, 2);
	assert.match(body, /RoPE · Rotate Q and K: query/);
	assert.match(body, /RoPE · Rotate Q and K: shared key/);
	for (const id of ['norm', 'key', 'value']) assert.match(body, new RegExp(`data-node="${id}"`));
	assert.match(body, /128D · bypass/);
	assert.match(body, /16 × \(128 \+ 64\)/);
	assert.match(body, /Latent 512/);
	assert.match(body, /\+ rotary K 64/);
	assert.match(body, /data-node="compression" aria-pressed="true"/);
});
test('MLA flow derives dimensions from the selected model', () => {
	const { body } = render(LatentFlow, {
		props: {
			hiddenSize: 4096,
			heads: 32,
			latentSize: 256,
			rotarySize: 32,
			contentSize: 64,
			valueSize: 64
		}
	});
	assert.match(body, /32 × \(64 \+ 32\)/);
	assert.match(body, /Latent 256/);
	assert.match(body, /\+ rotary K 32/);
	assert.match(body, /32 × 64 → 4096D/);
});
