import { recordedAssetUrl } from '$lib/recorded-json.js';
import { base } from '$app/paths';
import { get } from 'svelte/store';
import { hydrateRecordedExample } from '$lib/moe/adapter';
import type { TraceDocument, TraceExample } from '$lib/moe/types';
import {
	blockIdx,
	attentionHeadIdx,
	attentionHeadIdxTemp,
	moeTokenIndex,
	recordedControls,
	sampling,
	temperature
} from '~/store';
import { vitRecording, vitVisiblePositions, vitSelectedPosition, vitInspectorStage } from './store';
import type { VitDocument, VitRecording, TensorFile, TensorData } from './types';
import { tensor, row } from './types';

async function loadTensorFile(file: TensorFile): Promise<TensorData> {
	const response = await fetch(recordedAssetUrl(`${base}/traces/vit/${file.file}`));
	if (!response.ok) throw new Error(`ViT tensor ${file.file}: HTTP ${response.status}`);
	const bytes = await response.arrayBuffer();
	if (bytes.byteLength !== file.bytes) throw new Error(`Incomplete ViT recording ${file.file}`);
	const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)))
		.map((v) => v.toString(16).padStart(2, '0'))
		.join('');
	if (hash !== file.sha256) throw new Error(`ViT recording checksum failed: ${file.file}`);
	const values = new Float32Array(bytes);
	for (const entry of Object.values(file.tensors))
		if (entry.offset + entry.length > values.length)
			throw new Error(`Invalid ViT tensor bounds: ${file.file}`);
	return { descriptor: file, values };
}
function makeExample(recording: VitRecording, positions: number[]): TraceExample {
	const { document, common, layers } = recording,
		heads = document.model.attentionHeads,
		n = document.sequenceLength;
	const preview = (data: TensorData, name: string) => positions.map((p) => row(data, name, p, 8));
	const norms = (data: TensorData, name: string) =>
		positions.map((p) => Math.hypot(...row(data, name, p)));
	const headVectors = (data: TensorData, name: string) =>
		Array.from({ length: heads }, (_, h) => positions.map((p) => row(data, name, h * n + p, 8)));
	const headNorms = (data: TensorData, name: string) =>
		Array.from({ length: heads }, (_, h) =>
			positions.map((p) => Math.hypot(...row(data, name, h * n + p)))
		);
	const matrices = (name: string, scale = 1) =>
		layers.map((data) =>
			Array.from({ length: heads }, (_, h) =>
				positions.map((p) => {
					const values = tensor(data, name);
					return positions.map((k) => values[(h * n + p) * n + k] * scale);
				})
			)
		);
	const logits = Array.from(tensor(common, 'logits'));
	return {
		id: 'deit-colored-shapes-full',
		title: 'Colored shapes',
		category: 'Image classification',
		prompt: 'Colored shapes · 224 × 224 image',
		tokens: positions.map((p) => ({ index: p, id: p, text: document.tokens[p] })),
		layers: layers.map((data, index) => ({
			index,
			feedForwardType: 'dense',
			tokens: [],
			expertTokenCounts: [],
			hiddenInputPreview: preview(data, 'input'),
			residualInputPreview: preview(data, 'input'),
			residualAfterAttentionPreview: preview(data, 'residualAfterAttention'),
			feedForwardInputPreview: preview(data, 'norm2'),
			blockOutputPreview: preview(data, 'output'),
			attentionInputPreview: preview(data, 'norm1'),
			attentionInputNorms: norms(data, 'norm1'),
			attentionOutputPreview: preview(data, 'attentionOutput'),
			attentionOutputNorms: norms(data, 'attentionOutput'),
			queryPreview: headVectors(data, 'query'),
			keyPreview: headVectors(data, 'key'),
			valuePreview: headVectors(data, 'value'),
			queryNorms: headNorms(data, 'query'),
			keyNorms: headNorms(data, 'key'),
			valueNorms: headNorms(data, 'value')
		})),
		attention: matrices('attention'),
		attentionScores: matrices('scores', document.model.attentionMultiplier),
		attentionMaskedScores: matrices('scores', document.model.attentionMultiplier),
		embeddingPreview: preview(common, 'embedding'),
		embeddingNorms: norms(common, 'embedding'),
		finalHiddenPreview: preview(common, 'finalHidden'),
		finalHiddenNorms: norms(common, 'finalHidden'),
		nextTokenLogits: logits,
		nextTokenCandidates: [],
		validation: document.validation
	};
}
export function showVitPositions(
	positions: number[],
	selected = positions[0],
	preserve = true
): void {
	const recording = get(vitRecording);
	if (!recording) return;
	if (
		!positions.length ||
		new Set(positions).size !== positions.length ||
		positions.some((p) => p < 0 || p >= 197)
	)
		throw new Error('Invalid image position selection');
	const block = get(blockIdx),
		head = get(attentionHeadIdx);
	const example = makeExample(recording, positions);
	const document: TraceDocument = {
		schemaVersion: 1,
		model: { ...recording.document.model, outputTokenIndex: 0 },
		capture: recording.document.capture,
		examples: [example],
		vocabulary: recording.document.classLabels
	};
	// A classifier uses the full unmodified softmax, rather than token sampling.
	temperature.set(1);
	sampling.set({ type: 'top-p', value: 1 });
	vitVisiblePositions.set(positions);
	vitSelectedPosition.set(selected);
	hydrateRecordedExample(document, 0, { preserveSelection: preserve });
	moeTokenIndex.set(Math.max(0, positions.indexOf(selected)));
	if (preserve) {
		blockIdx.set(block);
		attentionHeadIdx.set(head);
		attentionHeadIdxTemp.set(head);
	}
}
/** Patches shown beside [CLS]: six rows in all, as many as Dense shows, so rows keep Dense's size. */
export const VIT_PATCH_PAGE = 5;
export const VIT_PATCH_PAGES = Math.ceil(196 / VIT_PATCH_PAGE);
export function showVitPatchPage(page: number): void {
	const first = Math.max(0, Math.min(VIT_PATCH_PAGES - 1, Math.floor(page))) * VIT_PATCH_PAGE + 1;
	showVitPositions(
		[0, ...Array.from({ length: Math.min(VIT_PATCH_PAGE, 197 - first) }, (_, i) => first + i)],
		first
	);
}
export async function loadVitRecording(): Promise<() => void> {
	const response = await fetch(recordedAssetUrl(`${base}/traces/vit/manifest.json`));
	if (!response.ok) throw new Error(`ViT manifest: HTTP ${response.status}`);
	const document = (await response.json()) as VitDocument;
	if (
		document.kind !== 'vit-full' ||
		document.layers.length !== 12 ||
		document.model.attentionHeads !== 3
	)
		throw new Error('Incomplete ViT model recording');
	const [common, ...layers] = await Promise.all([
		loadTensorFile(document.common),
		...document.layers.map(loadTensorFile)
	]);
	vitRecording.set({ document, common, layers });
	recordedControls.set({
		status: 'Recorded DeiT · all 12 encoder blocks · image classification',
		traceUrl: recordedAssetUrl(`${base}/traces/vit/manifest.json`),
		replay: () => showVitPositions(get(vitVisiblePositions), get(vitSelectedPosition))
	});
	showVitPositions(
		Array.from({ length: VIT_PATCH_PAGE + 1 }, (_, i) => i),
		0,
		false
	);
	return () => {
		vitRecording.set(null);
		vitInspectorStage.set(null);
		recordedControls.set(null);
	};
}
