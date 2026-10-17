import type { TraceDocument } from '$lib/moe/types';
export interface TensorEntry {
	offset: number;
	length: number;
	shape: number[];
}
export interface TensorFile {
	file: string;
	sha256: string;
	bytes: number;
	tensors: Record<string, TensorEntry>;
}
export interface VitDocument {
	schemaVersion: 1;
	kind: 'vit-full';
	model: TraceDocument['model'];
	sequenceLength: number;
	image: { src: string; alt: string; patchSize: number; gridSize: number; sha256: string };
	tokens: string[];
	classLabels: string[];
	common: TensorFile;
	layers: Array<TensorFile & { index: number }>;
	validation: Record<string, number>;
	capture: TraceDocument['capture'];
}
export interface TensorData {
	descriptor: TensorFile;
	values: Float32Array;
}
export interface VitRecording {
	document: VitDocument;
	common: TensorData;
	layers: TensorData[];
}
export function tensor(data: TensorData, name: string): Float32Array {
	const spec = data.descriptor.tensors[name];
	if (!spec) throw new Error(`Missing recorded tensor ${name}`);
	return data.values.subarray(spec.offset, spec.offset + spec.length);
}
export function row(data: TensorData, name: string, index: number, limit?: number): number[] {
	const spec = data.descriptor.tensors[name];
	if (!spec) throw new Error(`Missing recorded tensor ${name}`);
	const width = spec.shape.at(-1)!;
	return Array.from(
		tensor(data, name).subarray(index * width, index * width + Math.min(width, limit ?? width))
	);
}
export function scalar(data: TensorData, name: string, ...indices: number[]): number {
	const spec = data.descriptor.tensors[name];
	let offset = 0;
	for (let i = 0; i < indices.length; i++) offset = offset * spec.shape[i] + indices[i];
	return data.values[spec.offset + offset];
}
