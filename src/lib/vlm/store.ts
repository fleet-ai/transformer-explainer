import { writable, derived } from 'svelte/store';
import { blockIdx } from '~/store';
import { row } from '$lib/vit/types';
import type { VlmRecording, VlmPhase } from './types';
export const vlmRecording = writable<VlmRecording | null>(null);
export const vlmPhase = writable<VlmPhase>('vision');
export const vlmVisiblePositions = writable<number[]>([0, 1, 2, 3, 4, 5, 6, 7]);
export const vlmSelectedPosition = writable(0);
export const vlmInspectorStage = writable<string | null>(null);
export const vlmLoading = writable(false);
export const vlmError = writable('');
export const vlmLayer = derived(
	[vlmRecording, vlmPhase, blockIdx],
	([$recording, $phase, $block]) => $recording?.layers[$phase].get($block)
);
export const vlmHeadOperations = derived(
	[vlmRecording, vlmPhase, vlmVisiblePositions, blockIdx],
	([$recording, $phase, $positions, $block]) => {
		// Resolve phase and layer from the same update; a dependent layer store can briefly
		// still contain the previous stream while the phase has already changed.
		const $layer = $recording?.layers[$phase].get($block);
		if (!$recording || !$layer) return undefined;
		const { model, sequenceLength: n } = $recording.document.streams[$phase];
		if ($positions.some((p) => p < 0 || p >= n)) return undefined;
		const vectors = (name: string, heads: number) =>
			Array.from({ length: heads }, (_, h) => $positions.map((p) => row($layer, name, h * n + p)));
		return {
			queryBefore: vectors($phase === 'language' ? 'queryBefore' : 'query', model.attentionHeads),
			queryAfter: vectors('query', model.attentionHeads),
			keyBefore: vectors($phase === 'language' ? 'keyBefore' : 'key', model.keyValueHeads),
			keyAfter: vectors('key', model.keyValueHeads),
			value: vectors('value', model.keyValueHeads),
			attentionContext: vectors('context', model.attentionHeads)
		};
	}
);
