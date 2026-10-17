import type { TraceDocument, TraceExample } from '$lib/moe/types';
import type { AttentionOperationsDocument, AttentionOperationsLayer } from '$lib/moe/attention';
import type { ProjectionOperation } from '$lib/moe/operations';
export type Stack = 'encoder' | 'decoder';
export interface T5Attention extends AttentionOperationsLayer {
	scores: number[][][];
	maskedScores: number[][][];
	probabilities: number[][][];
	positionBias: number[][][];
}
export interface T5FeedForward {
	inputPreview: number[];
	up: ProjectionOperation;
	activation: number[];
	down: ProjectionOperation;
}
export interface T5Example {
	id: string;
	prompt: string;
	translation: string;
	generatedTokenIds: number[];
	encoder: { example: TraceExample; attentionOperations: AttentionOperationsDocument };
	decoder: { example: TraceExample; attentionOperations: AttentionOperationsDocument };
	crossAttention: T5Attention[];
	feedForward: Record<Stack, Array<{ index: number; tokens: T5FeedForward[] }>>;
	lmHead: {
		inputPreview: number[];
		unscaledInputPreview: number[];
		scale: number;
		candidateIds: number[];
		candidateLabels: string[];
		weights: number[][];
		omittedContribution: number[];
		logits: number[];
	};
}
export interface T5Document {
	schemaVersion: 1;
	kind: 'full-encoder-decoder';
	model: TraceDocument['model'];
	capture: TraceDocument['capture'];
	vocabulary: string[];
	examples: T5Example[];
	validation: Record<string, number>;
}
