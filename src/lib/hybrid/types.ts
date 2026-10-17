import type { TraceDocument } from '$lib/moe/types';
import type { TensorData, TensorFile } from '$lib/vit/types';
export interface HybridDocument {
	schemaVersion: 1;
	kind: 'hybrid-full';
	model: TraceDocument['model'] & {
		mambaHeads: number;
		mambaHeadSize: number;
		stateSize: number;
		stateChannelsRecorded: number;
		convolutionWidth: number;
		layerTypes: ('mamba' | 'hybrid')[];
		hybridLayers: number[];
	};
	prompt: string;
	tokens: string[];
	tokenIds: number[];
	sequenceLength: number;
	vocabulary: string[];
	common: TensorFile;
	layers: Array<TensorFile & { index: number; type: 'mamba' | 'hybrid' }>;
	nextTokenId: number;
	nextToken: string;
	validation: Record<string, number>;
	capture: TraceDocument['capture'];
}
export interface HybridRecording {
	document: HybridDocument;
	common: TensorData;
	layers: Map<number, TensorData>;
}
