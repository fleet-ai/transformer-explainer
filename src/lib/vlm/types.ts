import type { TraceDocument } from '$lib/moe/types';
import type { TensorFile, TensorData } from '$lib/vit/types';
export type VlmPhase = 'vision' | 'language';
export interface VlmStream {
	model: TraceDocument['model'];
	sequenceLength: number;
	tokens: string[];
	common: TensorFile;
	layers: Array<TensorFile & { index: number }>;
}
export interface VlmDocument {
	schemaVersion: 1;
	kind: 'vlm-full';
	model: TraceDocument['model'];
	streams: Record<VlmPhase, VlmStream>;
	prompt: string;
	formattedPrompt: string;
	tokenIds: number[];
	vocabulary: string[];
	imageTokenPositions: number[];
	connector: TensorFile;
	image: {
		src: string;
		alt: string;
		inputSize: number;
		patchSize: number;
		gridSize: number;
		pixelShuffleFactor: number;
	};
	completion: string;
	validation: Record<string, number>;
	capture: TraceDocument['capture'];
}
export interface VlmRecording {
	document: VlmDocument;
	common: Record<VlmPhase, TensorData>;
	connector: TensorData;
	layers: Record<VlmPhase, Map<number, TensorData>>;
}
