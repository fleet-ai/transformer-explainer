import { base } from '$app/paths';
import { get } from 'svelte/store';
import { hydrateRecordedExample } from '$lib/moe/adapter';
import type { TraceExample } from '$lib/moe/types';
import {
	blockIdx,
	attentionHeadIdx,
	attentionHeadIdxTemp,
	moeTokenIndex,
	recordedControls,
	moeExample,
	expandedBlock
} from '~/store';
import { row, tensor, type TensorData, type TensorFile } from '$lib/vit/types';
import type { HybridDocument, HybridRecording } from './types';
import { hybridRecording, hybridLoading, hybridError, hybridInspector } from './store';
const pending = new Map<number, Promise<TensorData>>();
let applying = false,
	disposed = false,
	generation = 0;
async function loadFile(file: TensorFile): Promise<TensorData> {
	const response = await fetch(`${base}/traces/hybrid/${file.file}`);
	if (!response.ok) throw new Error(`Hybrid recording ${file.file}: HTTP ${response.status}`);
	const bytes = await response.arrayBuffer();
	if (bytes.byteLength !== file.bytes) throw new Error(`Incomplete hybrid recording ${file.file}`);
	const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)))
		.map((x) => x.toString(16).padStart(2, '0'))
		.join('');
	if (digest !== file.sha256) throw new Error(`Hybrid recording checksum failed: ${file.file}`);
	return { descriptor: file, values: new Float32Array(bytes) };
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
function makeExample(recording: HybridRecording): TraceExample {
	const { document, common } = recording,
		n = document.sequenceLength,
		heads = document.model.attentionHeads;
	const positions = Array.from({ length: n }, (_, i) => i);
	const preview = (data: TensorData, name: string) => positions.map((p) => row(data, name, p, 8));
	const norms = (data: TensorData, name: string) =>
		positions.map((p) => Math.hypot(...row(data, name, p)));
	const headRows = (data: TensorData, name: string, norm = false) =>
		Array.from({ length: heads }, (_, h) =>
			positions.map((p) => {
				const values = row(data, name, h * n + p);
				return norm ? Math.hypot(...values) : values.slice(0, 8);
			})
		);
	const layers = document.layers.map((descriptor, index) => {
		const data = recording.layers.get(index);
		if (!data) return blankLayer(index);
		const commonLayer = {
			...blankLayer(index),
			residualInputPreview: preview(data, 'input'),
			hiddenInputPreview: preview(data, 'input'),
			blockOutputPreview: preview(data, 'output'),
			attentionInputPreview: preview(data, 'norm1'),
			attentionInputNorms: norms(data, 'norm1')
		};
		if (descriptor.type === 'mamba') return commonLayer;
		return {
			...commonLayer,
			attentionInputPreview: preview(data, 'attentionInput'),
			attentionInputNorms: norms(data, 'attentionInput'),
			attentionOutputPreview: preview(data, 'attentionOutput'),
			attentionOutputNorms: norms(data, 'attentionOutput'),
			feedForwardInputPreview: preview(data, 'ffnInput'),
			queryPreview: headRows(data, 'query') as number[][][],
			keyPreview: headRows(data, 'key') as number[][][],
			valuePreview: headRows(data, 'value') as number[][][],
			queryNorms: headRows(data, 'query', true) as number[][],
			keyNorms: headRows(data, 'key', true) as number[][],
			valueNorms: headRows(data, 'value', true) as number[][]
		};
	});
	const matrices = (name: string, scale = 1) =>
		document.layers.map((descriptor, index) => {
			const data = recording.layers.get(index);
			if (!data || descriptor.type !== 'hybrid') return [];
			const values = tensor(data, name);
			return Array.from({ length: heads }, (_, h) =>
				positions.map((p) => positions.map((k) => values[(h * n + p) * n + k] * scale))
			);
		});
	return {
		id: 'zamba2-full',
		title: 'The capital of France',
		category: 'Hybrid language model',
		prompt: document.prompt,
		tokens: document.tokens.map((text, index) => ({ text, index, id: document.tokenIds[index] })),
		layers,
		attention: matrices('attention'),
		attentionScores: matrices('scores', document.model.attentionMultiplier),
		attentionMaskedScores: matrices('scores', document.model.attentionMultiplier),
		embeddingPreview: preview(common, 'embedding'),
		embeddingNorms: norms(common, 'embedding'),
		finalHiddenPreview: preview(common, 'finalHidden'),
		finalHiddenNorms: norms(common, 'finalHidden'),
		nextTokenLogits: Array.from(tensor(common, 'logits')),
		nextTokenCandidates: [],
		validation: document.validation
	};
}
function applyCurrent() {
	const recording = get(hybridRecording);
	if (!recording) return;
	applying = true;
	const block = get(blockIdx),
		head = get(attentionHeadIdx),
		token = get(moeTokenIndex);
	hydrateRecordedExample({
		schemaVersion: 1,
		model: recording.document.model,
		capture: recording.document.capture,
		vocabulary: recording.document.vocabulary,
		examples: [makeExample(recording)]
	});
	blockIdx.set(block);
	attentionHeadIdx.set(head);
	attentionHeadIdxTemp.set(head);
	moeTokenIndex.set(Math.max(0, Math.min(recording.document.sequenceLength - 1, token)));
	applying = false;
}
async function ensureBlock(index: number) {
	const recording = get(hybridRecording);
	if (!recording) return;
	const request = ++generation;
	if (recording.layers.has(index)) {
		applyCurrent();
		hybridLoading.set(false);
		return;
	}
	hybridLoading.set(true);
	hybridError.set('');
	moeExample.update((example) =>
		example
			? {
					...example,
					layers: example.layers.map((layer, i) => (i === index ? blankLayer(i) : layer))
				}
			: example
	);
	try {
		if (!pending.has(index))
			pending.set(
				index,
				loadFile(recording.document.layers[index]).catch((error) => {
					pending.delete(index);
					throw error;
				})
			);
		const data = await pending.get(index)!;
		recording.layers.set(index, data);
		if (disposed || request !== generation) return;
		hybridRecording.set(recording);
		applyCurrent();
	} catch (reason) {
		if (!disposed && request === generation)
			hybridError.set(reason instanceof Error ? reason.message : String(reason));
	} finally {
		if (request === generation) hybridLoading.set(false);
	}
}
export async function selectHybridBlock(index: number) {
	const doc = get(hybridRecording)?.document;
	if (!doc) return;
	expandedBlock.set({ id: null });
	blockIdx.set(Math.max(0, Math.min(doc.layers.length - 1, index)));
}
export async function loadHybridRecording() {
	disposed = false;
	const response = await fetch(`${base}/traces/hybrid/manifest.json`);
	if (!response.ok) throw new Error(`Hybrid manifest: HTTP ${response.status}`);
	const document = (await response.json()) as HybridDocument;
	if (document.kind !== 'hybrid-full' || document.layers.length !== 38)
		throw new Error('Incomplete hybrid recording.');
	const common = await loadFile(document.common);
	hybridRecording.set({ document, common, layers: new Map() });
	applying = true;
	blockIdx.set(5);
	attentionHeadIdx.set(0);
	attentionHeadIdxTemp.set(0);
	moeTokenIndex.set(document.sequenceLength - 1);
	applying = false;
	recordedControls.set({
		status: 'Zamba2-1.2B · all 38 blocks · 6 shared-attention calls + 38 state-space mixers',
		traceUrl: `${base}/traces/hybrid/manifest.json`,
		replay: () => selectHybridBlock(5)
	});
	await ensureBlock(5);
	const unsubscribe = blockIdx.subscribe((index) => {
		if (!applying && !disposed) void ensureBlock(index);
	});
	return () => {
		disposed = true;
		generation++;
		unsubscribe();
		hybridRecording.set(null);
		hybridInspector.set(null);
		recordedControls.set(null);
		pending.clear();
	};
}
