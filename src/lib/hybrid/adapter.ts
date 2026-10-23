import { recordedAssetUrl } from '$lib/recorded-json.js';
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
import {
	hybridRecording,
	hybridLoading,
	hybridError,
	hybridInspector,
	hybridStage,
	hybridMambaView
} from './store';
const pending = new Map<number, Promise<TensorData>>();
let applying = false,
	disposed = false,
	generation = 0;
async function loadFile(file: TensorFile): Promise<TensorData> {
	const response = await fetch(recordedAssetUrl(`${base}/traces/hybrid/${file.file}`));
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
	const token = get(moeTokenIndex);
	const model = recording.document.model;
	// The Mamba view stacks the 64 SSM heads where the attention view stacks its 32 heads.
	const heads = get(hybridMambaView) ? model.mambaHeads : model.attentionHeads;
	const head = Math.min(heads - 1, get(attentionHeadIdx));
	// Keep the block selected: resetting it here would notify the block subscriber again after
	// `applying` is cleared (Svelte queues nested store updates) and bounce between two blocks.
	hydrateRecordedExample(
		{
			schemaVersion: 1,
			model: { ...model, attentionHeads: heads },
			capture: recording.document.capture,
			vocabulary: recording.document.vocabulary,
			examples: [makeExample(recording)]
		},
		0,
		{ preserveSelection: true }
	);
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
export async function selectHybridBlock(
	index: number,
	stage: 'transformer' | 'mamba' = 'transformer'
) {
	const doc = get(hybridRecording)?.document;
	if (!doc) return;
	expandedBlock.set({ id: null });
	const next = Math.max(0, Math.min(doc.layers.length - 1, index));
	const changed = next !== get(blockIdx);
	applying = true;
	hybridStage.set(doc.layers[next].type === 'hybrid' ? stage : 'mamba');
	applying = false;
	if (changed) blockIdx.set(next);
	else void ensureBlock(next);
}
/** Step through the computation: a hybrid block is two steps, its transformer and then its mixer. */
export function stepHybrid(delta: 1 | -1) {
	const doc = get(hybridRecording)?.document;
	if (!doc) return;
	const steps = doc.layers.flatMap((layer) =>
		layer.type === 'hybrid'
			? [
					{ index: layer.index, stage: 'transformer' as const },
					{ index: layer.index, stage: 'mamba' as const }
				]
			: [{ index: layer.index, stage: 'mamba' as const }]
	);
	const block = get(blockIdx),
		stage = doc.layers[block].type === 'hybrid' ? get(hybridStage) : 'mamba';
	const at = steps.findIndex((step) => step.index === block && step.stage === stage);
	const target = steps[Math.max(0, Math.min(steps.length - 1, at + delta))];
	void selectHybridBlock(target.index, target.stage);
}
/** Whether the step buttons can move further in this direction. */
export function canStepHybrid(delta: 1 | -1, block: number, stage: 'transformer' | 'mamba') {
	const doc = get(hybridRecording)?.document;
	if (!doc) return false;
	const hybrid = doc.layers[block]?.type === 'hybrid';
	if (delta < 0) return block > 0 || (hybrid && stage === 'mamba');
	return block < doc.layers.length - 1 || (hybrid && stage === 'transformer');
}
export async function loadHybridRecording() {
	disposed = false;
	const response = await fetch(recordedAssetUrl(`${base}/traces/hybrid/manifest.json`));
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
		traceUrl: recordedAssetUrl(`${base}/traces/hybrid/manifest.json`),
		replay: () => selectHybridBlock(5)
	});
	await ensureBlock(5);
	const unsubscribe = blockIdx.subscribe((index) => {
		if (!applying && !disposed) void ensureBlock(index);
	});
	const unsubscribeStage = hybridStage.subscribe(() => {
		if (!applying && !disposed) void ensureBlock(get(blockIdx));
	});
	return () => {
		disposed = true;
		generation++;
		unsubscribe();
		unsubscribeStage();
		hybridStage.set('transformer');
		hybridRecording.set(null);
		hybridInspector.set(null);
		recordedControls.set(null);
		pending.clear();
	};
}
