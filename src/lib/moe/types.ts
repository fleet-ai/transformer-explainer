export interface TokenTrace {
	routerLogits: number[];
	routerProbabilities: number[];
	selectedExperts: number[];
	routingWeights: number[];
	expertOutputNorms: number[];
	weightedOutputNorms: number[];
	expertOutputPreview: number[][];
	combinedOutputPreview: number[];
	/** Routed experts and the always-active shared branch, before residual addition. */
	routedCombinedOutputPreview?: number[];
	sharedExpertOutputPreview?: number[];
	inputNorm: number;
	outputNorm: number;
	reconstructionMaxAbsError: number;
}

export interface TraceToken {
	index: number;
	id: number;
	text: string;
}

export interface TraceExample {
	id: string;
	title: string;
	category: string;
	prompt: string;
	tokens: TraceToken[];
	layers: Array<{
		index: number;
		feedForwardType?: 'dense' | 'moe';
		hiddenInputPreview?: number[][];
		residualInputPreview?: number[][];
		residualAfterAttentionPreview?: number[][];
		feedForwardInputPreview?: number[][];
		blockOutputPreview?: number[][];
		tokens: TokenTrace[];
		expertTokenCounts: number[];
		attentionInputPreview: number[][];
		attentionInputNorms: number[];
		attentionOutputPreview: number[][];
		attentionOutputNorms: number[];
		queryPreview: number[][][];
		keyPreview: number[][][];
		valuePreview: number[][][];
		queryNorms: number[][];
		keyNorms: number[][];
		valueNorms: number[][];
	}>;
	attention: number[][][][];
	attentionScores: number[][][][];
	attentionMaskedScores: number[][][][];
	embeddingPreview: number[][];
	embeddingNorms: number[];
	finalHiddenPreview: number[][];
	finalHiddenNorms: number[];
	nextTokenLogits: number[];
	nextTokenCandidates: Array<{ id: number; text: string; probability: number }>;
	validation: Record<string, unknown>;
}

export interface TraceDocument {
	schemaVersion: number;
	model: {
		architecture?: 'moe' | 'mla' | 'vit' | 'encoder-decoder' | 'vision-language' | 'hybrid';
		maskMode?: 'causal' | 'bidirectional';
		attentionBias?: 'relative';
		encoderLayers?: number;
		decoderLayers?: number;
		visionLayers?: number;
		outputTokenIndex?: number;
		stack?: 'encoder' | 'decoder';
		activation?: string;
		normalization?: 'rmsnorm' | 'layernorm';
		outputKind?: 'language' | 'classification';
		id: string;
		name: string;
		revision: string;
		url: string;
		license: string;
		numLayers: number;
		numExperts: number;
		topK: number;
		hiddenSize: number;
		intermediateSize: number;
		totalParameters: number;
		activeParametersApprox: number;
		residualMultiplier?: number;
		attentionHeads: number;
		keyValueHeads: number;
		attentionHeadSize: number;
		contentKeySize?: number;
		rotarySize?: number;
		valueHeadSize?: number;
		kvLatentSize?: number;
		sharedExperts?: number;
		denseLayers?: number;
		denseIntermediateSize?: number;
		sharedIntermediateSize?: number;
		normalizeRoutingWeights?: boolean;
		rmsNormEps?: number;
		attentionMultiplier: number;
		embeddingMultiplier: number;
		logitsScaling: number;
		ropeTheta: number;
		vocabSize: number;
		attentionMaskValue: number;
	};
	capture: {
		timestamp: string;
		device: string;
		dtype: string;
		python: string;
		packages: Record<string, string>;
		script: string;
		notes: string | string[];
	};
	examples: TraceExample[];
	vocabulary: string[];
}
