export type ArchitectureId = 'mla' | 'encoder-decoder' | 'hybrid' | 'vision-language' | 'vit';

export interface RecordedTensor {
	name: string;
	shape: number[];
	values: number[] | number[][];
	rowLabels?: string[];
	columnLabels?: string[];
	slice?: string;
	role?: string;
	head?: number;
}

export interface RecordedOperation {
	id: string;
	label: string;
	kind: string;
	description: string;
	formula: string;
	tensors: RecordedTensor[];
	metrics?: { label: string; value: number | string; unit?: string }[];
}

export interface RecordedLayer {
	index: number;
	label?: string;
	operations: RecordedOperation[];
}

export interface RecordedExample {
	id: string;
	label: string;
	input: {
		text?: string;
		image?:
			| string
			| {
					src?: string;
					url?: string;
					alt?: string;
					width?: number;
					height?: number;
					patchSize?: number;
			  };
	};
	tokens: string[];
	output?: { text?: string; predictions?: { label: string; probability: number }[] };
	layers: RecordedLayer[];
}

export interface ArchitectureTrace {
	schemaVersion: number;
	architecture: ArchitectureId;
	title: string;
	provenance: {
		modelId: string;
		revision: string;
		weightsSha256?: string | Record<string, unknown>;
		libraryVersions?: Record<string, string>;
		captureScript: string;
		scope: string;
		trained: boolean;
		partial?: boolean;
	};
	config: Record<string, unknown>;
	examples: RecordedExample[];
}

export function tensorRows(tensor?: RecordedTensor): number[][] {
	if (!tensor?.values.length) return [];
	if (Array.isArray(tensor.values[0])) return tensor.values as number[][];
	const flat = tensor.values as number[];
	if (tensor.rowLabels?.length === flat.length) return flat.map((value) => [value]);
	const width =
		tensor.shape.length === 2 && tensor.shape[0] * tensor.shape[1] === flat.length
			? tensor.shape[1]
			: flat.length;
	return Array.from({ length: Math.ceil(flat.length / width) }, (_, index) =>
		flat.slice(index * width, (index + 1) * width)
	);
}

export function numeric(value: number): string {
	if (!Number.isFinite(value)) return String(value);
	return value !== 0 && (Math.abs(value) < 0.00001 || Math.abs(value) >= 100000)
		? value.toExponential(5)
		: value.toFixed(6);
}

/** Resolve a logical position to its saved row, without confusing repeated tokens or sparse slices. */
export function tensorRowForPosition(
	tensor: RecordedTensor | undefined,
	position: number,
	label = ''
): number {
	if (!tensor) return -1;
	const rows = tensorRows(tensor);
	if (tensor.rowLabels && label) {
		if (tensor.rowLabels[position] === label) return position;
		return tensor.rowLabels.indexOf(label);
	}
	if (tensor.shape.length > 1 && rows.length < tensor.shape[0]) return -1;
	return position >= 0 && position < rows.length ? position : -1;
}

export function tensorColor(value: number, scale: number, probability = false): string {
	if (!Number.isFinite(value)) return '#f3f0f7';
	const strength = Math.min(1, Math.abs(value) / (scale || 1));
	return value < 0 && !probability
		? `rgba(83, 143, 217, ${0.07 + strength * 0.75})`
		: `rgba(137, 85, 207, ${0.07 + strength * 0.85})`;
}

export function isArchitectureTrace(
	value: unknown,
	architecture: ArchitectureId
): value is ArchitectureTrace {
	const trace = value as ArchitectureTrace;
	return (
		!!trace &&
		trace.schemaVersion === 1 &&
		trace.architecture === architecture &&
		!!trace.provenance?.modelId &&
		Array.isArray(trace.examples) &&
		trace.examples.length > 0 &&
		trace.examples.every(
			(example) =>
				Array.isArray(example.tokens) &&
				example.layers?.length > 0 &&
				example.layers.every((layer) => Array.isArray(layer.operations))
		)
	);
}
