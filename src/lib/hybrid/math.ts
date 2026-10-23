import { row, scalar, type TensorData } from '$lib/vit/types';

export const silu = (x: number) => x / (1 + Math.exp(-x));
export const softplus = (x: number) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
export const dot = (a: number[], b: number[], start = 0) =>
	a.reduce((sum, value, i) => sum + (i >= start ? value * b[i] : 0), 0);

/** Reconstruct one causal depthwise convolution from the four saved kernel taps. */
export function convolution(data: TensorData, token: number, channel: number) {
	const width = data.descriptor.tensors.convWeights.shape[2];
	const terms = Array.from({ length: width }, (_, k) => {
		const position = token - width + 1 + k;
		const input = position < 0 ? 0 : scalar(data, 'convInput', position, channel);
		const weight = scalar(data, 'convWeights', channel, 0, k);
		return { position, input, weight, product: input * weight };
	});
	const bias = scalar(data, 'convBias', channel);
	const reconstructed = terms.reduce((sum, t) => sum + t.product, bias);
	return {
		terms,
		bias,
		reconstructed,
		raw: scalar(data, 'convRaw', token, channel),
		output: scalar(data, 'convOutput', token, channel)
	};
}

/** The saved state is the capture's sequential reconstruction, not a kernel state hook. */
export function stateStep(
	data: TensorData,
	token: number,
	head: number,
	channel: number,
	state: number
) {
	const dtRaw = scalar(data, 'dtRaw', token, head),
		bias = scalar(data, 'dtBias', head);
	const dt = scalar(data, 'dt', token, head),
		A = scalar(data, 'A', head);
	const x = scalar(data, 'x', token, head, channel),
		B = scalar(data, 'B', token, state);
	const C = scalar(data, 'C', token, state),
		D = scalar(data, 'D', head);
	const previous = token === 0 ? 0 : scalar(data, 'state', token - 1, head, channel, state);
	const decay = Math.exp(dt * A),
		injection = dt * B * x;
	const saved = scalar(data, 'state', token, head, channel, state);
	return {
		dtRaw,
		bias,
		dt,
		A,
		B,
		C,
		D,
		x,
		previous,
		decay,
		injection,
		saved,
		dtReconstructed: Math.max(0.001, softplus(dtRaw + bias)),
		reconstructed: previous * decay + injection,
		contribution: saved * C
	};
}

export function readout(data: TensorData, token: number, head: number, channel: number) {
	const size = data.descriptor.tensors.state.shape[3];
	const terms = Array.from(
		{ length: size },
		(_, s) => scalar(data, 'state', token, head, channel, s) * scalar(data, 'C', token, s)
	);
	const skip = scalar(data, 'D', head) * scalar(data, 'x', token, head, channel);
	const index = head * data.descriptor.tensors.x.shape[2] + channel;
	return {
		terms,
		skip,
		reconstructed: terms.reduce((a, b) => a + b, skip),
		sequential: scalar(data, 'sequentialReadout', token, index),
		recorded: scalar(data, 'scanOutput', token, index)
	};
}

export function gatedNorm(data: TensorData, token: number, index: number) {
	const scan = row(data, 'scanOutput', token),
		gate = row(data, 'gate', token);
	const products = scan.map((x, i) => x * silu(gate[i]));
	const meanSquare = products.reduce((a, b) => a + b * b, 0) / products.length;
	const weight = scalar(data, 'gatedNormWeight', index),
		denominator = Math.sqrt(meanSquare + 1e-5);
	return {
		scan: scan[index],
		gate: gate[index],
		activation: silu(gate[index]),
		product: products[index],
		meanSquare,
		denominator,
		weight,
		reconstructed: (products[index] / denominator) * weight,
		recorded: scalar(data, 'gated', token, index)
	};
}

export function projection(
	data: TensorData,
	token: number,
	weightsName: string,
	inputName: string,
	outputName: string,
	outputIndex: number
) {
	const input = row(data, inputName, token),
		weights = row(data, weightsName, outputIndex);
	return {
		input,
		weights,
		first: dot(weights.slice(0, 4), input),
		remaining: dot(weights, input, 4),
		reconstructed: dot(weights, input),
		output: scalar(data, outputName, token, outputIndex)
	};
}

export function ropePair(
	data: TensorData,
	common: TensorData,
	token: number,
	head: number,
	coordinate: number,
	kind: 'query' | 'key'
) {
	const size = data.descriptor.tensors[kind].shape[2],
		partner = coordinate + size / 2;
	const a = scalar(data, kind + 'Before', head, token, coordinate),
		b = scalar(data, kind + 'Before', head, token, partner);
	const cos = scalar(common, 'cos', token, coordinate),
		sin = scalar(common, 'sin', token, coordinate);
	return {
		coordinate,
		partner,
		a,
		b,
		cos,
		sin,
		angle: Math.atan2(sin, cos),
		afterA: scalar(data, kind, head, token, coordinate),
		afterB: scalar(data, kind, head, token, partner),
		reconstructedA: a * cos - b * sin,
		reconstructedB: b * cos + a * sin
	};
}

/**
 * Mamba-2's scan written as one causal token-mixing matrix per head (its "SSD" form):
 * y[t] = Σ_{s≤t} M[t,s] · x[s] + D · x[t], with M[t,s] = L[t,s] · (C[t] · B[s]) and the decay
 * L[t,s] = Δ[s] · exp(A · Σ_{k=s+1..t} Δ[k]). Every entry comes from recorded B, C, Δ and A;
 * the upper triangle is masked (NaN) because no token reads a later one.
 */
export function mixingMatrix(data: TensorData, head: number) {
	const n = data.descriptor.tensors.B.shape[0];
	const B = Array.from({ length: n }, (_, s) => row(data, 'B', s));
	const C = Array.from({ length: n }, (_, t) => row(data, 'C', t));
	const dt = Array.from({ length: n }, (_, t) => scalar(data, 'dt', t, head));
	const A = scalar(data, 'A', head);
	const scores = C.map((c) => B.map((b) => dot(c, b)));
	const decay = scores.map((_, t) =>
		scores.map((__, s) => {
			if (s > t) return NaN;
			let total = 0;
			for (let k = s + 1; k <= t; k++) total += dt[k];
			return dt[s] * Math.exp(A * total);
		})
	);
	const mixing = decay.map((values, t) => values.map((l, s) => (s > t ? NaN : l * scores[t][s])));
	return { scores, decay, mixing, A, dt, D: scalar(data, 'D', head) };
}
