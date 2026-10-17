import type { TokenTrace } from './types';

export function percent(value: number, digits = 1): string {
	return `${(value * 100).toFixed(digits)}%`;
}

export function tokenLabel(text: string): string {
	return text.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '⇥') || '∅';
}

export function parameterCount(value: number): string {
	return value >= 1e9 ? `${(value / 1e9).toFixed(2)}B` : `${Math.round(value / 1e6)}M`;
}

export function expertLoads(tokens: TokenTrace[], count: number): number[] {
	const loads = Array<number>(count).fill(0);
	for (const token of tokens) {
		for (const expert of token.selectedExperts) loads[expert] += 1;
	}
	return loads;
}
