import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

function load(path, dependencies = {}) {
	const exports = {};
	const source = ts.transpileModule(readFileSync(new URL(path, import.meta.url), 'utf8'), {
		compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
	}).outputText;
	runInNewContext(source, {
		exports,
		require: (name) => {
			assert.ok(dependencies[name], `Unexpected dependency ${name}`);
			return dependencies[name];
		}
	});
	return exports;
}
const tensors = load('../src/lib/vit/types.ts');
const math = load('../src/lib/hybrid/math.ts', { '$lib/vit/types': tensors });
const root = new URL('../static/traces/hybrid/', import.meta.url);
const document = JSON.parse(readFileSync(new URL('manifest.json', root)));
function data(descriptor) {
	const bytes = readFileSync(new URL(descriptor.file, root));
	return {
		descriptor,
		values: new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
	};
}
const near = (actual, expected, tolerance = 3e-5) =>
	assert.ok(
		Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
		`${actual} != ${expected}`
	);

test('inspector reconstructs actual causal convolutions, selected state, scan readout and gated RMSNorm', () => {
	for (const descriptor of document.layers) {
		const layer = data(descriptor);
		for (const token of [0, document.sequenceLength - 1]) {
			for (const head of [0, 17, 63])
				for (const channel of [0, 3]) {
					const conv = math.convolution(layer, token, head * 64 + channel);
					near(conv.reconstructed, conv.raw);
					near(math.silu(conv.raw), conv.output);
					const scan = math.readout(layer, token, head, channel);
					near(scan.reconstructed, scan.sequential);
					near(scan.reconstructed, scan.recorded, 5e-4);
					for (const coordinate of [0, 31, 127]) {
						const state = math.stateStep(layer, token, head, channel, coordinate);
						near(state.dtReconstructed, state.dt, 2e-6);
						near(state.reconstructed, state.saved, 3e-6);
						if (token === 0) assert.equal(state.previous, 0);
					}
					const gated = math.gatedNorm(layer, token, head * 64 + channel);
					near(gated.reconstructed, gated.recorded);
				}
			for (const channel of [4096, 4223, 4224, 4351]) {
				const conv = math.convolution(layer, token, channel);
				near(conv.reconstructed, conv.raw);
			}
			for (const output of [0, 3])
				for (const names of [
					['inWeights', 'norm1', 'mambaProjection'],
					['outWeights', 'gated', 'mambaOutput']
				]) {
					const p = math.projection(layer, token, ...names, output);
					near(p.reconstructed, p.output);
					near(p.first + p.remaining, p.output);
				}
		}
	}
});

test('inspector uses all 128 rotary dimensions and real adapted FFN matrices in every hybrid block', () => {
	const common = data(document.common);
	for (const descriptor of document.layers.filter((x) => x.type === 'hybrid')) {
		const layer = data(descriptor);
		for (const token of [0, document.sequenceLength - 1]) {
			for (let head = 0; head < 32; head++)
				for (let coordinate = 0; coordinate < 64; coordinate++) {
					for (const kind of ['query', 'key']) {
						const r = math.ropePair(layer, common, token, head, coordinate, kind);
						assert.equal(r.partner, coordinate + 64);
						near(r.reconstructedA, r.afterA, 2e-6);
						near(r.reconstructedB, r.afterB, 2e-6);
						near(r.cos ** 2 + r.sin ** 2, 1, 2e-6);
					}
				}
			for (const names of [
				['gateWeights', 'ffnInput', 'gateFFN'],
				['upWeights', 'ffnInput', 'up'],
				['downWeights', 'product', 'down']
			])
				for (let output = 0; output < 4; output++) {
					const p = math.projection(layer, token, ...names, output);
					near(p.first + p.remaining, p.output);
				}
		}
	}
});
