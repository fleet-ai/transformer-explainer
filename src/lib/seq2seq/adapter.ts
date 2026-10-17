import { recordedAssetUrl } from '$lib/recorded-json.js';
import { base } from '$app/paths';
import { readRecordedJson } from '$lib/recorded-json.js';
import { get } from 'svelte/store';
import { hydrateRecordedExample } from '$lib/moe/adapter';
import {
	blockIdx,
	attentionHeadIdx,
	attentionHeadIdxTemp,
	moeAttentionOperations,
	recordedControls
} from '~/store';
import { t5Document, t5ExampleIndex, t5Stack, t5Inspector } from './store';
import type { T5Document, Stack } from './types';
export function selectT5Example(index: number, stack: Stack = get(t5Stack), preserveBlock = false) {
	const document = get(t5Document);
	if (!document || !document.examples[index]) return;
	const previousBlock = get(blockIdx),
		previousHead = get(attentionHeadIdx);
	const model = {
		...document.model,
		stack,
		maskMode: stack === 'encoder' ? ('bidirectional' as const) : ('causal' as const)
	};
	hydrateRecordedExample(
		{
			schemaVersion: 1,
			model,
			capture: document.capture,
			vocabulary: document.vocabulary,
			examples: document.examples.map((e) => e[stack].example)
		},
		index
	);
	moeAttentionOperations.set(document.examples[index][stack].attentionOperations);
	t5ExampleIndex.set(index);
	t5Stack.set(stack);
	t5Inspector.set(null);
	if (preserveBlock) {
		blockIdx.set(previousBlock);
		attentionHeadIdx.set(previousHead);
		attentionHeadIdxTemp.set(previousHead);
	}
}
export function selectT5Stack(stack: Stack) {
	selectT5Example(get(t5ExampleIndex), stack, true);
}
export async function loadT5Recording() {
	const response = await fetch(recordedAssetUrl(`${base}/traces/t5-full.json`));
	if (!response.ok) throw new Error(`T5 recording returned HTTP ${response.status}.`);
	const document = (await readRecordedJson(response)) as T5Document;
	if (
		document.kind !== 'full-encoder-decoder' ||
		document.examples.some(
			(e) => e.encoder.example.layers.length !== 6 || e.decoder.example.layers.length !== 6
		)
	)
		throw new Error('Incomplete T5 recording.');
	t5Document.set(document);
	selectT5Example(0, 'decoder');
	recordedControls.set({
		status: 'T5-small · full 6 encoder + 6 decoder blocks · all 8 heads',
		traceUrl: recordedAssetUrl(`${base}/traces/t5-full.json`),
		selectExample: (index) => selectT5Example(index),
		replay: () => selectT5Example(get(t5ExampleIndex))
	});
	return () => {
		recordedControls.set(null);
		t5Inspector.set(null);
		t5Document.set(null);
	};
}
