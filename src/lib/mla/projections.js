/** The QKV bank contains recorded projections before head-local RoPE.
 * Rotary coordinates in queryBefore/keyBefore use the trace's paired ordering.
 * Never pad absent recordings with schematic values.
 * @param {import('../moe/attention').AttentionOperationsLayer | undefined} layer
 * @param {number} token
 */
export function mlaBankVectors(layer, token) {
	if (!layer) return {};
	/** @param {number[][][] | undefined} content
	 * @param {number[][][] | undefined} rotary
	 */
	const combine = (content, rotary, shared = false) => {
		if (!content || !rotary) return undefined;
		const values = [];
		for (let head = 0; head < content.length; head++) {
			const c = content[head][token],
				r = rotary[shared ? 0 : head]?.[token];
			if (!c || !r) return undefined;
			values.push(...c, ...r);
		}
		return values;
	};
	return {
		query: combine(layer.queryContent, layer.queryBefore),
		key: combine(layer.keyContent, layer.keyBefore, true),
		value: layer.value?.every((rows) => rows[token])
			? layer.value.flatMap((rows) => rows[token])
			: undefined
	};
}

/** Shapes and captured inputs/outputs for the selected token and head.
 * @param {import('../moe/attention').AttentionOperationsLayer | undefined} layer
 * @param {import('../moe/types').TraceDocument['model'] | null | undefined} model
 * @param {number} head
 * @param {number} token
 * @param {string} stage
 */
export function mlaProjectionView(layer, model, head, token, stage) {
	const hidden = model?.hiddenSize ?? 2048,
		heads = model?.attentionHeads ?? 16;
	const content = model?.contentKeySize ?? 128,
		rotary = model?.rotarySize ?? 64;
	const latent = model?.kvLatentSize ?? 512,
		value = model?.valueHeadSize ?? 128;
	/** @param {string} label @param {number[] | undefined} values */
	const vector = (label, values, color = 'latent') => ({ label, values, color });
	const x = layer?.attentionInput?.[token],
		raw = layer?.rawLatent?.[token],
		normalized = layer?.kvLatent?.[token];
	const k = layer?.keyContent?.[head]?.[token],
		v = layer?.value?.[head]?.[token];
	/** @param {'queryProjection' | 'attentionContext' | 'keyContent' | 'value'} field */
	const flatten = (field) => {
		const rows = layer?.[field];
		return rows?.every((head) => head[token]) ? rows.flatMap((head) => head[token]) : undefined;
	};
	if (stage === 'query')
		return {
			label: 'Query projection',
			input: 'RMSNorm(x)',
			weight: 'Wqᵀ',
			output: 'Q · all heads',
			inputSize: hidden,
			outputSize: heads * (content + rotary),
			inputValues: x,
			outputValues: flatten('queryProjection'),
			vectors: [vector('Q · all heads · before RoPE', flatten('queryProjection'), 'query')]
		};
	if (stage === 'compression')
		return {
			label: 'KV compression',
			input: 'RMSNorm(x)',
			weight: 'W↓KVᵀ',
			output: 'Raw latent + rotary K',
			inputSize: hidden,
			outputSize: latent + rotary,
			inputValues: x,
			outputValues:
				raw && layer?.rawKeyRotary?.[0]?.[token]
					? [...raw, ...layer.rawKeyRotary[0][token]]
					: undefined,
			vectors: [
				vector('Raw KV latent', raw),
				vector('Shared rotary K · before RoPE', layer?.keyBefore?.[0]?.[token], 'key')
			]
		};
	if (stage === 'norm')
		return {
			label: 'Latent RMSNorm',
			input: 'Raw KV latent',
			weight: '',
			output: 'Normalized latent',
			inputSize: latent,
			outputSize: latent,
			activation: 'RMSNorm',
			inputValues: raw,
			outputValues: normalized,
			vectors: [vector('Raw KV latent', raw), vector('Normalized KV latent', normalized)]
		};
	if (stage === 'key' || stage === 'value') {
		const isKey = stage === 'key';
		return {
			label: `${isKey ? 'Key' : 'Value'} expansion`,
			input: 'Normalized latent',
			weight: isKey ? 'W↑Kᵀ' : 'W↑Vᵀ',
			output: isKey ? 'K content · all heads' : 'V · all heads',
			inputSize: latent,
			outputSize: heads * (isKey ? content : value),
			inputValues: normalized,
			outputValues: flatten(isKey ? 'keyContent' : 'value'),
			vectors: [
				vector('Shared normalized KV latent', normalized),
				vector(
					isKey ? 'K content · all heads' : 'V · all heads',
					flatten(isKey ? 'keyContent' : 'value'),
					isKey ? 'key' : 'value'
				)
			]
		};
	}
	if (stage === 'expansion')
		return {
			label: `K / V expansion · head ${head + 1}`,
			input: 'Normalized latent',
			weight: 'W↑KVᵀ',
			output: 'K content + V',
			inputSize: latent,
			outputSize: content + value,
			inputValues: normalized,
			outputValues: k && v ? [...k, ...v] : undefined,
			vectors: [
				vector('Shared normalized KV latent', normalized),
				vector(`K content · head ${head + 1}`, k, 'key'),
				vector(`V · head ${head + 1}`, v, 'value')
			]
		};
	return {
		label: 'Attention output',
		input: 'Concatenated heads',
		weight: 'Woutᵀ',
		output: 'Attention output',
		inputSize: heads * value,
		outputSize: hidden,
		inputValues: flatten('attentionContext'),
		outputValues: layer?.attentionOutput?.[token],
		vectors: [
			vector(
				`Attention-weighted V · head ${head + 1}`,
				layer?.attentionContext?.[head]?.[token],
				'value'
			),
			vector('Projected attention output', layer?.attentionOutput?.[token])
		]
	};
}
