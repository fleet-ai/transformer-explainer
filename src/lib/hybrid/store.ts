import { createInspector } from '$lib/inspectors';
import { writable, derived } from 'svelte/store';
import { blockIdx } from '~/store';
import type { HybridRecording } from './types';
import { row } from '$lib/vit/types';
export const hybridRecording = writable<HybridRecording | null>(null);
export const hybridLayer = derived(
	[hybridRecording, blockIdx],
	([$recording, $block]) => $recording?.layers.get($block) ?? null
);
export const hybridInspector = createInspector<string | null>();
export const hybridLoading = writable(false);
export const hybridError = writable('');
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
