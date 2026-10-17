export interface PromptCompletion {
	text: string;
	token: string;
	error: string;
	finished: boolean;
}

/** Append tokenizer-decoded text, never the probability list's display label. */
export function resolvePromptCompletion(
	prompt: string,
	vocabulary: readonly string[] | undefined,
	tokenId: number | undefined
): PromptCompletion {
	const unchanged = { text: prompt, token: '', error: '', finished: false };
	if (tokenId === undefined || !Number.isInteger(tokenId) || tokenId < 0) return unchanged;
	const token = vocabulary?.[tokenId];
	if (typeof token !== 'string') return unchanged;
	if (token === '<|endoftext|>') return { ...unchanged, finished: true };
	if (token.includes('\uFFFD')) {
		return {
			...unchanged,
			error: 'Cannot safely append this partial Unicode token. Edit the prompt to continue.'
		};
	}
	return { ...unchanged, text: prompt + token, token };
}

/** An edited draft replaces the preview; an unchanged prompt accepts it once. */
export function preparePromptSubmission(
	draft: string,
	completion: PromptCompletion,
	edited: boolean
): string {
	return edited ? draft : completion.text;
}
