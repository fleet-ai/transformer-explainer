import { createInspector } from '$lib/inspectors';
import { writable, derived } from 'svelte/store';
import { attentionHeadIdx, blockIdx } from '~/store';
import type { HybridRecording } from './types';
import { row } from '$lib/vit/types';
import { mixingMatrix } from './math';
export const hybridRecording = writable<HybridRecording | null>(null);
export const hybridLayer = derived(
	[hybridRecording, blockIdx],
	([$recording, $block]) => $recording?.layers.get($block) ?? null
);
export const hybridInspector = createInspector<string | null>();
export const hybridLoading = writable(false);
export const hybridError = writable('');
/** A hybrid block runs the shared transformer, then a Mamba mixer; the page shows one half. */
export const hybridStage = writable<'transformer' | 'mamba'>('transformer');
/** True when the page shows a Mamba mixer: every Mamba block, and the second half of a hybrid one. */
export const hybridMambaView = derived(
	[hybridRecording, blockIdx, hybridStage],
	([$recording, $block, $stage]) =>
		!!$recording && ($recording.document.layers[$block]?.type === 'mamba' || $stage === 'mamba')
);
export const hybridHeadOperations = derived(
	[hybridRecording, hybridLayer],
	([$recording, $layer]) => {
		if (!$recording || !$layer || !$layer.descriptor.tensors.query) return undefined;
		const n = $recording.document.sequenceLength,
			heads = $recording.document.model.attentionHeads;
		const vectors = (name: string) =>
			Array.from({ length: heads }, (_, h) =>
				Array.from({ length: n }, (_, t) => row($layer, name, h * n + t))
			);
		return {
			queryBefore: vectors('queryBefore'),
			keyBefore: vectors('keyBefore'),
			queryAfter: vectors('query'),
			keyAfter: vectors('key'),
			value: vectors('value'),
			attentionContext: vectors('context')
		};
	}
);
/** Recorded tensors of the selected SSM head, laid out for the head card. */
export const hybridMambaOperations = derived(
	[hybridRecording, hybridLayer, attentionHeadIdx],
	([$recording, $layer, $head]) => {
		if (!$recording || !$layer) return undefined;
		const model = $recording.document.model;
		const n = $recording.document.sequenceLength;
		const head = Math.min(model.mambaHeads - 1, Math.max(0, $head));
		const tokens = Array.from({ length: n }, (_, t) => t);
		const slice = (name: string, t: number) =>
			row($layer, name, t).slice(head * model.mambaHeadSize, (head + 1) * model.mambaHeadSize);
		return {
			head,
			B: tokens.map((t) => row($layer, 'B', t)),
			C: tokens.map((t) => row($layer, 'C', t)),
			x: tokens.map((t) => row($layer, 'x', t * model.mambaHeads + head)),
			y: tokens.map((t) => slice('scanOutput', t)),
			/** The in_proj output in its real order: z, then x, B and C before the conv, then raw Δ. */
			z: tokens.map((t) => row($layer, 'gate', t)),
			xbc: tokens.map((t) => row($layer, 'convInput', t)),
			dtRaw: tokens.map((t) => row($layer, 'dtRaw', t)),
			gated: tokens.map((t) => row($layer, 'gated', t)),
			scan: tokens.map((t) => row($layer, 'scanOutput', t)),
			mixer: tokens.map((t) => row($layer, 'mambaOutput', t)),
			matrix: mixingMatrix($layer, head)
		};
	}
);
/** The operation under the pointer; its bands and boxes light up together. */
export const hybridHot = writable<string | null>(null);
