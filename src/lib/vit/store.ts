import { createInspector } from '$lib/inspectors';
import { writable, derived } from 'svelte/store';
import { blockIdx } from '~/store';
import type { VitRecording } from './types';
import { row } from './types';
export const vitRecording = writable<VitRecording | null>(null);
export const vitVisiblePositions = writable<number[]>([0, 1, 2, 3, 4, 5]);
export const vitSelectedPosition = writable(0);
export const vitInspectorStage = createInspector<string | null>();
/** Full recorded vectors for visible positions; no omitted token is removed from inference. */
export const vitHeadOperations = derived(
	[vitRecording, vitVisiblePositions, blockIdx],
	([$recording, $positions, $block]) => {
		if (!$recording) return undefined;
		const data = $recording.layers[$block];
		if (!data) return undefined;
		const n = $recording.document.sequenceLength,
			heads = $recording.document.model.attentionHeads;
		const vectors = (name: string) =>
			Array.from({ length: heads }, (_, h) => $positions.map((p) => row(data, name, h * n + p)));
		return {
			queryBefore: vectors('query'),
			queryAfter: vectors('query'),
			keyBefore: vectors('key'),
			keyAfter: vectors('key'),
			value: vectors('value'),
			attentionContext: vectors('context')
		};
	}
);
