import { base } from '$app/paths';
import { traceAssetUrl, trimLiveRecordings } from './live';

export interface ProjectionOperation {
	weights: number[][];
	omittedContribution: number[];
	output: number[];
}

export interface ExpertOperation {
	id: number;
	/** Dense and fused shared experts can have different intermediate widths. */
	intermediateSize?: number;
	gate: ProjectionOperation;
	up: ProjectionOperation;
	siluGate: number[];
	product: number[];
	down: ProjectionOperation;
	weightedOutput: number[];
}

export interface ExpertOperationsDocument {
	schemaVersion: number;
	kind: 'expert-operations';
	model: {
		id: string;
		revision: string;
		hiddenSize: number;
		intermediateSize: number;
		topK: number;
	};
	example: { id: string; prompt: string };
	layers: Array<{
		index: number;
		tokens: Array<{
			inputPreview: number[];
			selectedExperts: number[];
			routingWeights: number[];
			experts: ExpertOperation[];
			combinedOutput: number[];
			routedCombinedOutput?: number[];
			sharedExpertOutput?: number[];
		}>;
	}>;
}

const recordings = new Map<string, Promise<ExpertOperationsDocument>>();

/** Load arithmetic only when an expert is inspected; reuse it across layers/tokens. */
export function loadExpertOperations(exampleId: string): Promise<ExpertOperationsDocument> {
	let recording = recordings.get(exampleId);
	if (!recording) {
		recording = fetch(traceAssetUrl(exampleId, 'expert', base))
			.then(async (response) => {
				if (!response.ok) throw new Error(`Expert recording returned HTTP ${response.status}.`);
				const document: ExpertOperationsDocument = await response.json();
				if (document.kind !== 'expert-operations' || document.example.id !== exampleId) {
					throw new Error('The expert recording does not match this prompt.');
				}
				return document;
			})
			.catch((error) => {
				recordings.delete(exampleId);
				throw error;
			});
		recordings.set(exampleId, recording);
		trimLiveRecordings(recordings);
	}
	return recording;
}
