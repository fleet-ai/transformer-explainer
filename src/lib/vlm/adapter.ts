import { base } from '$app/paths';
import { get } from 'svelte/store';
import { hydrateRecordedExample, recordedProbabilities } from '$lib/moe/adapter';
import type { TraceDocument, TraceExample } from '$lib/moe/types';
import {
	blockIdx,
	attentionHeadIdx,
	attentionHeadIdxTemp,
	moeTokenIndex,
	recordedControls,
	moeExample,
	moeModel,
	modelData,
	expandedBlock
} from '~/store';
import { row, tensor, type TensorData, type TensorFile } from '$lib/vit/types';
import type { VlmDocument, VlmPhase, VlmRecording } from './types';
import {
	vlmRecording,
	vlmPhase,
	vlmVisiblePositions,
	vlmSelectedPosition,
	vlmLoading,
	vlmError,
	vlmInspectorStage
} from './store';
const pending = new Map<string, Promise<TensorData>>();
let generation = 0,
	applying = false,
	disposed = false;
async function loadFile(directory: string, file: TensorFile): Promise<TensorData> {
	const key = `${directory}/${file.file}`;
	if (!pending.has(key))
		pending.set(
			key,
			(async () => {
				const response = await fetch(`${base}/traces/vlm/${key}`);
				if (!response.ok) throw new Error(`VLM recording ${key}: HTTP ${response.status}`);
				const bytes = await response.arrayBuffer();
				if (bytes.byteLength !== file.bytes) throw new Error(`Incomplete recording ${key}`);
				const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)))
					.map((x) => x.toString(16).padStart(2, '0'))
					.join('');
				if (digest !== file.sha256) throw new Error(`Recording checksum failed: ${key}`);
				return { descriptor: file, values: new Float32Array(bytes) };
			})().catch((error) => {
				pending.delete(key);
				throw error;
			})
		);
	return pending.get(key)!;
}
function blankLayer(index: number): TraceExample['layers'][number] {
	return {
		index,
		feedForwardType: 'dense',
		tokens: [],
		expertTokenCounts: [],
		attentionInputPreview: [],
		attentionInputNorms: [],
		attentionOutputPreview: [],
		attentionOutputNorms: [],
		queryPreview: [],
		keyPreview: [],
		valuePreview: [],
		queryNorms: [],
		keyNorms: [],
		valueNorms: []
	};
}
function makeExample(recording: VlmRecording, phase: VlmPhase, positions: number[]): TraceExample {
	const stream = recording.document.streams[phase],
		n = stream.sequenceLength,
		model = stream.model;
	const preview = (data: TensorData, name: string) => positions.map((p) => row(data, name, p, 8));
	const norms = (data: TensorData, name: string) =>
		positions.map((p) => Math.hypot(...row(data, name, p)));
	const headRows = (data: TensorData, name: string, norm = false) =>
		Array.from({ length: model.attentionHeads }, (_, h) =>
			positions.map((p) => {
				const actualHead =
					name === 'query' ? h : Math.floor(h / (model.attentionHeads / model.keyValueHeads));
				const values = row(data, name, actualHead * n + p);
				return norm ? Math.hypot(...values) : values.slice(0, 8);
			})
		);
	const layers = stream.layers.map((_, index) => {
		const data = recording.layers[phase].get(index);
		if (!data) return blankLayer(index);
		return {
			...blankLayer(index),
			hiddenInputPreview: preview(data, 'input'),
			residualInputPreview: preview(data, 'input'),
			residualAfterAttentionPreview: preview(data, 'residualAfterAttention'),
			feedForwardInputPreview: preview(data, 'norm2'),
			blockOutputPreview: preview(data, 'output'),
			attentionInputPreview: preview(data, 'norm1'),
			attentionInputNorms: norms(data, 'norm1'),
			attentionOutputPreview: preview(data, 'attentionOutput'),
			attentionOutputNorms: norms(data, 'attentionOutput'),
			queryPreview: headRows(data, 'query') as number[][][],
			keyPreview: headRows(data, 'key') as number[][][],
			valuePreview: headRows(data, 'value') as number[][][],
			queryNorms: headRows(data, 'query', true) as number[][],
			keyNorms: headRows(data, 'key', true) as number[][],
			valueNorms: headRows(data, 'value', true) as number[][]
		};
	});
	const matrices = (name: string, scale = 1) =>
		stream.layers.map((_, index) => {
			const data = recording.layers[phase].get(index);
			if (!data) return [];
			const values = tensor(data, name);
			return Array.from({ length: model.attentionHeads }, (_, h) =>
				positions.map((p) => positions.map((k) => values[(h * n + p) * n + k] * scale))
			);
		});
	const common = recording.common[phase],
		logits = Array.from(tensor(recording.common.language, 'logits'));
	return {
		id: `smolvlm-full-${phase}`,
		title: 'Colored shapes',
		category: 'Vision-language',
		prompt: recording.document.prompt,
		tokens: positions.map((p) => ({
			index: p,
			id: phase === 'language' ? recording.document.tokenIds[p] : p,
			text: stream.tokens[p]
		})),
		layers,
		attention: matrices('attention'),
		attentionScores: matrices('scores', model.attentionMultiplier),
		attentionMaskedScores: matrices('scores', model.attentionMultiplier),
		embeddingPreview: preview(common, 'embedding'),
		embeddingNorms: norms(common, 'embedding'),
		finalHiddenPreview: preview(common, 'finalHidden'),
		finalHiddenNorms: norms(common, 'finalHidden'),
		nextTokenLogits: logits,
		nextTokenCandidates: [],
		validation: recording.document.validation
	};
}
function applyCurrent(): void {
	const recording = get(vlmRecording);
	if (!recording) return;
	applying = true;
	const phase = get(vlmPhase),
		positions = get(vlmVisiblePositions),
		head = get(attentionHeadIdx);
	const stream = recording.document.streams[phase],
		example = makeExample(recording, phase, positions);
	const model = {
		...stream.model,
		outputTokenIndex: Math.max(0, positions.indexOf(stream.sequenceLength - 1))
	};
	hydrateRecordedExample(
		{
			schemaVersion: 1,
			model,
			capture: recording.document.capture,
			examples: [example],
			vocabulary: recording.document.vocabulary
		},
		0,
		{ preserveSelection: true }
	);
	attentionHeadIdx.set(Math.min(head, model.attentionHeads - 1));
	attentionHeadIdxTemp.set(Math.min(head, model.attentionHeads - 1));
	moeTokenIndex.set(Math.max(0, positions.indexOf(get(vlmSelectedPosition))));
	applying = false;
}
async function ensureBlock(index: number): Promise<void> {
	const recording = get(vlmRecording);
	if (!recording) return;
	const phase = get(vlmPhase),
		request = ++generation;
	if (recording.layers[phase].has(index)) {
		applyCurrent();
		vlmLoading.set(false);
		return;
	}
	vlmLoading.set(true);
	vlmError.set('');
	// Clear the active data while loading, rather than showing another block's tensors.
	moeExample.update((example) =>
		example
			? {
					...example,
					layers: example.layers.map((layer, i) => (i === index ? blankLayer(i) : layer))
				}
			: example
	);
	try {
		const data = await loadFile(phase, recording.document.streams[phase].layers[index]);
		recording.layers[phase].set(index, data);
		if (disposed || request !== generation) return;
		vlmRecording.set(recording);
		applyCurrent();
	} catch (reason) {
		if (!disposed && request === generation)
			vlmError.set(reason instanceof Error ? reason.message : String(reason));
	} finally {
		if (request === generation) vlmLoading.set(false);
	}
}
export async function selectVlmPhase(phase: VlmPhase): Promise<void> {
	const recording = get(vlmRecording);
	if (!recording) return;
	generation++;
	applying = true;
	vlmPhase.set(phase);
	blockIdx.set(0);
	attentionHeadIdx.set(0);
	attentionHeadIdxTemp.set(0);
	expandedBlock.set({ id: null });
	const n = recording.document.streams[phase].sequenceLength,
		start = phase === 'language' ? Math.floor((n - 1) / 8) * 8 : 0;
	vlmVisiblePositions.set(Array.from({ length: Math.min(8, n - start) }, (_, i) => start + i));
	vlmSelectedPosition.set(phase === 'language' ? n - 1 : 0);
	vlmInspectorStage.set(null);
	applying = false;
	// Populate this stream's metadata immediately, then fetch only its selected block.
	applyCurrent();
	await ensureBlock(0);
}
export function selectVlmPage(page: number): void {
	const recording = get(vlmRecording);
	if (!recording) return;
	const phase = get(vlmPhase),
		n = recording.document.streams[phase].sequenceLength;
	const start = Math.max(0, Math.min(Math.ceil(n / 8) - 1, Math.floor(page))) * 8;
	const positions = Array.from({ length: Math.min(8, n - start) }, (_, i) => start + i);
	if (phase === 'language' && !positions.includes(n - 1)) positions.push(n - 1);
	vlmVisiblePositions.set(positions);
	vlmSelectedPosition.set(start);
	applyCurrent();
}
export async function loadVlmRecording(): Promise<() => void> {
	disposed = false;
	const response = await fetch(`${base}/traces/vlm/manifest.json`);
	if (!response.ok) throw new Error(`VLM manifest: HTTP ${response.status}`);
	const document = (await response.json()) as VlmDocument;
	if (
		document.kind !== 'vlm-full' ||
		document.streams.vision.layers.length !== 12 ||
		document.streams.language.layers.length !== 30
	)
		throw new Error('Incomplete VLM recording');
	const [vision, language, connector] = await Promise.all([
		loadFile('vision', document.streams.vision.common),
		loadFile('language', document.streams.language.common),
		loadFile('', document.connector)
	]);
	vlmRecording.set({
		document,
		common: { vision, language },
		connector,
		layers: { vision: new Map(), language: new Map() }
	});
	recordedControls.set({
		status: 'Recorded SmolVLM · 12 vision + 30 language blocks',
		traceUrl: `${base}/traces/vlm/manifest.json`,
		replay: () => selectVlmPhase(get(vlmPhase))
	});
	await selectVlmPhase('vision');
	const unsubscribe = blockIdx.subscribe((index) => {
		if (!applying && !disposed) void ensureBlock(index);
	});
	return () => {
		disposed = true;
		generation++;
		unsubscribe();
		vlmRecording.set(null);
		vlmInspectorStage.set(null);
		recordedControls.set(null);
		pending.clear();
	};
}
