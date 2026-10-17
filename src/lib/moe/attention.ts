import { base } from '$app/paths';
import type { TraceToken } from './types';
import { traceAssetUrl, trimLiveRecordings } from './live';

export interface AttentionOperationsLayer {
	index: number;
	/** Head × token × all head coordinates; K/V have unique shared heads. */
	queryBefore: number[][][];
	queryAfter: number[][][];
	keyBefore: number[][][];
	keyAfter: number[][][];
	value: number[][][];
}

export interface AttentionOperationsDocument {
	schemaVersion: 1;
	kind: 'attention-operations';
	model: {
		id: string;
		revision: string;
		attentionHeads: number;
		keyValueHeads: number;
		headSize: number;
		ropeTheta: number;
		attentionMultiplier: number;
	};
	capture: {
		timestamp: string;
		script: string;
		scriptSha256: string;
		mainBundleSha256: string;
		mainCaptureScriptSha256: string;
		device: string;
		dtype: string;
		python: string;
		packages: Record<string, string>;
		matchesMainVocabularyLogits: boolean;
		actualValueUnchangedByRotation: boolean;
		actualGroupedKeyValueRepetitionExact: boolean;
		projectionHeadReshapeExact: boolean;
		notes: string[];
	};
	example: { id: string; prompt: string; tokens: TraceToken[] };
	positions: number[];
	/** Token × headSize/2. The model's identical second halves are omitted. */
	rotary: { cos: number[][]; sin: number[][] };
	layers: AttentionOperationsLayer[];
	validation: Record<string, unknown>;
}

export interface HeadMapping {
	queryHead: number;
	keyValueHead: number;
	sharedQueryHeads: number[];
}

/** Independent pretrained Qwen recording; never relabeled as a Granite head. */
export interface RopeExampleDocument {
	schemaVersion: 1;
	kind: 'rope-example';
	model: {
		id: string;
		name: string;
		url: string;
		revision: string;
		headSize: number;
		attentionHeads: number;
		keyValueHeads: number;
		hiddenSize: number;
		numLayers: number;
		ropeTheta: number;
		ropeScaling: null;
		attentionMultiplier: number;
		license: string;
	};
	capture: {
		timestamp: string;
		script: string;
		scriptSha256: string;
		device: string;
		dtype: string;
		python: string;
		packages: Record<string, string>;
		modelSourceSha256: string;
		configSha256: string;
		hookedVsUnhookedMaxLogitError: number;
		actualValueUnchangedByRotation: boolean;
		actualNormalizationOutputsMatchRotaryInputs: boolean;
		actualGroupedKeyValueRepetitionExact: boolean;
		queryInputNormalization: 'q_norm';
		keyInputNormalization: 'k_norm';
		notes: string[];
	};
	example: { id: string; prompt: string; tokens: TraceToken[] };
	scope: { layer: number; queryHead: number; keyValueHead: number };
	positions: number[];
	rotary: { cos: number[][]; sin: number[][] };
	/** Token × all 128 coordinates; before Q/K are after Qwen's q_norm/k_norm. */
	queryBefore: number[][];
	queryAfter: number[][];
	keyBefore: number[][];
	keyAfter: number[][];
	value: number[][];
	validation: Record<string, unknown>;
}

/** All indices are zero-based. Contiguous query groups share one K/V head. */
export function headMapping(queryHead: number, queryHeads = 16, keyValueHeads = 8): HeadMapping {
	if (![queryHeads, keyValueHeads].every((count) => Number.isInteger(count) && count > 0)
		|| queryHeads % keyValueHeads !== 0) {
		throw new RangeError('Query head count must be a positive multiple of key/value head count.');
	}
	if (!Number.isInteger(queryHead) || queryHead < 0 || queryHead >= queryHeads) {
		throw new RangeError('Query head is outside the recorded head range.');
	}
	const groupSize = queryHeads / keyValueHeads;
	const keyValueHead = Math.floor(queryHead / groupSize);
	return {
		queryHead,
		keyValueHead,
		sharedQueryHeads: Array.from({ length: groupSize }, (_, index) => keyValueHead * groupSize + index)
	};
}

export interface RotaryPair {
	indices: [number, number];
	before: [number, number];
	cosine: number;
	sine: number;
	reconstructed: [number, number];
	recorded: [number, number];
	/** Signed browser reconstruction minus the captured float32 result. */
	difference: [number, number];
}

export interface RotaryCoordinate {
	coordinate: number;
	partner: number;
	pairIndex: number;
}

/** Map either half to its canonical pair, preserving the rotation's signs. */
export function ropeCoordinate(coordinate: number, headSize = 64): RotaryCoordinate {
	if (!Number.isInteger(headSize) || headSize < 2 || headSize % 2 !== 0) {
		throw new RangeError('RoPE head size must be a positive even integer.');
	}
	if (!Number.isInteger(coordinate) || coordinate < 0 || coordinate >= headSize) {
		throw new RangeError('RoPE coordinate is outside the head range.');
	}
	const half = headSize / 2;
	return {
		coordinate,
		partner: coordinate < half ? coordinate + half : coordinate - half,
		pairIndex: coordinate % half
	};
}

/** Inspect the model's half-split rotation pair (i, i + headSize/2). */
export function ropePair(
	before: number[], after: number[], cos: number[], sin: number[], pairIndex: number
): RotaryPair {
	const half = before.length / 2;
	if (!Number.isInteger(half) || half < 1 || after.length !== before.length
		|| cos.length !== half || sin.length !== half) {
		throw new RangeError('RoPE requires equal even-sized head vectors and half-sized cosine/sine arrays.');
	}
	if (!Number.isInteger(pairIndex) || pairIndex < 0 || pairIndex >= half) {
		throw new RangeError('RoPE pair is outside the captured head range.');
	}
	const second = pairIndex + half;
	const a = before[pairIndex];
	const b = before[second];
	const cosine = cos[pairIndex];
	const sine = sin[pairIndex];
	const reconstructed: [number, number] = [a * cosine - b * sine, b * cosine + a * sine];
	const recorded: [number, number] = [after[pairIndex], after[second]];
	const difference: [number, number] = [reconstructed[0] - recorded[0], reconstructed[1] - recorded[1]];
	if (![a, b, cosine, sine, ...recorded, ...reconstructed, ...difference].every(Number.isFinite)) {
		throw new Error('RoPE coordinates and arithmetic must be finite.');
	}
	return { indices: [pairIndex, second], before: [a, b], cosine, sine, reconstructed, recorded, difference };
}

/** Four distinct coordinates: the selected pair and an adjacent frequency pair. */
export function ropePairWindow(
	before: number[], after: number[], cos: number[], sin: number[], coordinate: number
): [RotaryPair, RotaryPair] {
	const selected = ropeCoordinate(coordinate, before.length);
	if (before.length < 4) throw new RangeError('Two RoPE pairs require at least four head coordinates.');
	// Keep the last pair visible without wrapping, duplicating, or leaving the head.
	const firstPair = Math.min(selected.pairIndex, before.length / 2 - 2);
	return [
		ropePair(before, after, cos, sin, firstPair),
		ropePair(before, after, cos, sin, firstPair + 1)
	];
}

function isFiniteVector(value: unknown, length: number): value is number[] {
	return Array.isArray(value) && value.length === length && value.every(Number.isFinite);
}

function checkRecording(value: unknown, exampleId: string): AttentionOperationsDocument {
	const data = value as AttentionOperationsDocument;
	if (data?.schemaVersion !== 1 || data.kind !== 'attention-operations' || data.example?.id !== exampleId) {
		throw new Error('The attention recording does not match this prompt or schema.');
	}
	const { attentionHeads, keyValueHeads, headSize } = data.model ?? {};
	headMapping(0, attentionHeads, keyValueHeads);
	const count = data.example.tokens?.length;
	if (!Number.isInteger(headSize) || headSize < 2 || headSize % 2 !== 0 || !count
		|| !isFiniteVector(data.positions, count) || !Array.isArray(data.layers) || !data.layers.length
		|| !Array.isArray(data.rotary?.cos) || !Array.isArray(data.rotary?.sin)
		|| data.rotary.cos.length !== count || data.rotary.sin.length !== count
		|| !data.rotary.cos.every((row) => isFiniteVector(row, headSize / 2))
		|| !data.rotary.sin.every((row) => isFiniteVector(row, headSize / 2))) {
		throw new Error('The attention recording has invalid rotary dimensions or values.');
	}
	for (const [index, layer] of data.layers.entries()) {
		if (layer.index !== index) throw new Error('The attention recording has misaligned layers.');
		for (const field of ['queryBefore', 'queryAfter', 'keyBefore', 'keyAfter', 'value'] as const) {
			const heads = field.startsWith('query') ? attentionHeads : keyValueHeads;
			if (!Array.isArray(layer[field]) || layer[field].length !== heads
				|| !layer[field].every((head) => Array.isArray(head) && head.length === count
					&& head.every((row) => isFiniteVector(row, headSize)))) {
				throw new Error(`The attention recording has invalid ${field} dimensions or values.`);
			}
		}
	}
	return data;
}

const recordings = new Map<string, Promise<AttentionOperationsDocument>>();

/** Fetch full head arithmetic on demand; failures are evicted so retries can succeed. */
export function loadAttentionOperations(exampleId: string): Promise<AttentionOperationsDocument> {
	let recording = recordings.get(exampleId);
	if (!recording) {
		recording = fetch(traceAssetUrl(exampleId, 'attention', base))
			.then(async (response) => {
				if (!response.ok) throw new Error(`Attention recording returned HTTP ${response.status}.`);
				return checkRecording(await response.json(), exampleId);
			})
			.catch((error) => {
				recordings.delete(exampleId);
				throw error;
			});
		recordings.set(exampleId, recording);
		trimLiveRecordings(recordings);
	}
	return recording;
}

function checkRopeExample(value: unknown): RopeExampleDocument {
	const data = value as RopeExampleDocument;
	if (data?.schemaVersion !== 1 || data.kind !== 'rope-example'
		|| data.example?.id !== 'qwen3-128' || data.model?.headSize !== 128) {
		throw new Error('The RoPE example does not match the recorded 128-dimensional model.');
	}
	const { headSize, attentionHeads, keyValueHeads, numLayers } = data.model;
	if (![attentionHeads, keyValueHeads, numLayers].every((count) => Number.isInteger(count) && count > 0)) {
		throw new Error('The RoPE example has invalid model dimensions.');
	}
	const scope = data.scope;
	if (!scope || !Number.isInteger(scope.layer) || scope.layer < 0 || scope.layer >= numLayers
		|| headMapping(scope.queryHead, attentionHeads, keyValueHeads).keyValueHead !== scope.keyValueHead) {
		throw new Error('The RoPE example has an invalid layer or head selection.');
	}
	const count = data.example.tokens?.length;
	if (!count || !isFiniteVector(data.positions, count)) {
		throw new Error('The RoPE example has invalid token positions.');
	}
	for (const field of ['queryBefore', 'queryAfter', 'keyBefore', 'keyAfter', 'value'] as const) {
		if (!Array.isArray(data[field]) || data[field].length !== count
			|| !data[field].every((row) => isFiniteVector(row, headSize))) {
			throw new Error(`The RoPE example has invalid ${field} dimensions or values.`);
		}
	}
	for (const field of ['cos', 'sin'] as const) {
		if (!Array.isArray(data.rotary?.[field]) || data.rotary[field].length !== count
			|| !data.rotary[field].every((row) => isFiniteVector(row, headSize / 2))) {
			throw new Error('The RoPE example has invalid rotary dimensions or values.');
		}
	}
	return data;
}

let ropeExample: Promise<RopeExampleDocument> | undefined;

/** Lazy-load the separate 128D model example, independently of the active Granite trace. */
export function loadRopeExample(): Promise<RopeExampleDocument> {
	if (!ropeExample) {
		ropeExample = fetch(`${base}/traces/rope-example-128.json`)
			.then(async (response) => {
				if (!response.ok) throw new Error(`RoPE example returned HTTP ${response.status}.`);
				return checkRopeExample(await response.json());
			})
			.catch((error) => {
				ropeExample = undefined;
				throw error;
			});
	}
	return ropeExample;
}
