import { writable, derived } from 'svelte/store';
import type { T5Document, Stack } from './types';
export const t5Document = writable<T5Document | null>(null);
export const t5ExampleIndex = writable(0);
export const t5Stack = writable<Stack>('decoder');
export const t5Inspector = writable<'cross' | 'ffn' | 'output' | 'self' | null>(null);
export const t5Example = derived(
	[t5Document, t5ExampleIndex],
	([$document, $index]) => $document?.examples[$index] ?? null
);
