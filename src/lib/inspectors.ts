import { get, writable, type Writable } from 'svelte/store';

/** One inspector owns the foreground, including transitions between model-specific panels. */
export function createInspectorGroup() {
	let closeActive: (() => void) | undefined;
	return function createInspector<T = unknown>(): Writable<T | null> {
		const state = writable<T | null>(null);
		const close = () => state.set(null);
		function set(value: T | null) {
			if (value != null) {
				if (closeActive !== close) closeActive?.();
				closeActive = close;
			} else if (closeActive === close) {
				closeActive = undefined;
			}
			state.set(value);
		}
		return { subscribe: state.subscribe, set, update: (fn) => set(fn(get(state))) };
	};
}

export const createInspector = createInspectorGroup();
export const attentionInspector = createInspector<
	{ kind: 'heads' } | { kind: 'rope'; vector: 'query' | 'key' }
>();
