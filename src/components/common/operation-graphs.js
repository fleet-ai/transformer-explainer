// UI topology only. Tensor values remain owned by each recorded-model inspector.
/**
 * @typedef {{id:string,x:number,y:number,label:string,dim:string,action:string,terminal?:boolean}} BandNode
 * @typedef {{from:string,to:string,action:string,label?:string,via?:number,color?:string}} BandEdge
 * @typedef {{hiddenSize?:number,outputSize?:number,headSize?:number,heads?:number,kvHeads?:number,sequenceLength?:number,normalization?:string,causal?:boolean,relativeBias?:boolean,scale?:string,rotary?:boolean,label?:string}} GraphOptions
 * @param {string} kind
 * @param {GraphOptions} p
 */
export function operationGraph(kind, p = {}) {
	const {
		hiddenSize = 512,
		outputSize = hiddenSize,
		headSize = 64,
		heads = 8,
		kvHeads = heads,
		sequenceLength = 6,
		normalization = 'RMSNorm',
		causal = false,
		relativeBias = false,
		scale = '',
		rotary = false,
		label = 'Projection'
	} = p;
	/** @type {(id:string,x:number,y:number,text:string,dim?:string,action?:string)=>BandNode} */
	const node = (id, x, y, text, dim = '', action = id) => ({ id, x, y, label: text, dim, action });
	/** @type {(id:string,x:number,y:number,text:string,dim:string)=>BandNode} */
	const terminal = (id, x, y, text, dim) => ({ ...node(id, x, y, text, dim), terminal: true });
	/** @type {(from:string,to:string,action:string,label?:string,via?:number,color?:string)=>BandEdge} */
	const edge = (from, to, action, label, via, color) => ({ from, to, action, label, via, color });
	/** @param {BandNode[]} nodes @param {BandEdge[]} edges */
	const graph = (nodes, edges, height = 164, caption = '', boxWidth = 0.16) => ({
		nodes,
		edges,
		height,
		caption,
		boxWidth
	});
	if (kind === 'qkv')
		return graph(
			[
				terminal('input', 0, 116, 'Normalized X', `${hiddenSize}D`),
				...['query', 'key', 'value'].map((id, i) =>
					node(
						id,
						0.43,
						48 + i * 68,
						['Wq', 'Wk', 'Wv'][i],
						`${hiddenSize} → ${(i === 0 ? heads : kvHeads) * headSize}`,
						id
					)
				),
				...['query', 'key', 'value'].map((id, i) =>
					terminal(
						id + 'Out',
						1,
						48 + i * 68,
						['Q', 'K', 'V'][i],
						`${i === 0 ? heads : kvHeads} × ${headSize}D`
					)
				)
			],
			['query', 'key', 'value'].flatMap((id, i) => [
				edge('input', id, id),
				edge(id, id + 'Out', id, undefined, undefined, ['#c3dafa', '#f2cbd0', '#c5e7d9'][i])
			]),
			228,
			'Three projections from the same input'
		);
	if (kind === 'attention')
		return graph(
			[
				terminal('q', 0, 72, rotary ? 'Q′' : 'Q', `${headSize}D`),
				terminal('k', 0, 157, rotary ? 'K′' : 'K', `${sequenceLength} × ${headSize}`),
				node('scores', 0.23, 72, 'QKᵀ', `${sequenceLength} scores`),
				node(
					'softmax',
					0.5,
					72,
					'Softmax',
					relativeBias
						? causal
							? 'bias + mask'
							: 'relative bias'
						: causal
							? `${scale} · mask`
							: scale
				),
				terminal('v', 0.49, 157, 'V', `${sequenceLength} × ${headSize}`),
				node('context', 0.77, 72, 'A × V', `${headSize}D`),
				terminal('out', 1, 72, 'Context', `${headSize}D`)
			],
			[
				edge('q', 'scores', 'scores'),
				edge('k', 'scores', 'scores'),
				edge('scores', 'softmax', 'softmax'),
				edge('softmax', 'context', 'context'),
				edge('v', 'context', 'context'),
				edge('context', 'out', 'context')
			],
			204,
			'Per head · Q/K scores → probabilities → weighted V',
			0.16
		);
	if (kind === 'mamba')
		return graph(
			[
				terminal('input', 0, 126, 'Token', '2048D'),
				node('projection', 0.17, 126, 'Win', '2048 → 8512'),
				node('convolution', 0.39, 126, 'Conv + SiLU', 'x/B/C · 4352'),
				node('scan', 0.6, 126, 'State + readout', '4096D'),
				node('gate', 0.79, 126, 'Gate + RMSNorm', '4096D'),
				node('output', 0.96, 126, 'Wout', '2048D'),
				node('z', 0.48, 48, 'SiLU(z)', 'gate · 4096', 'gate'),
				node('dt', 0.4, 211, 'softplus(Δ)', '64 heads', 'scan')
			],
			[
				edge('input', 'projection', 'projection'),
				edge('projection', 'convolution', 'convolution'),
				edge('convolution', 'scan', 'scan'),
				edge('scan', 'gate', 'gate'),
				edge('gate', 'output', 'output'),
				edge('projection', 'z', 'projection'),
				edge('z', 'gate', 'gate'),
				edge('projection', 'dt', 'scan'),
				edge('dt', 'scan', 'scan')
			],
			252,
			'State also carries sₜ₋₁ → sₜ · output joins the residual',
			0.14
		);
	if (kind === 'rope')
		return graph(
			[
				terminal('q', 0, 54, 'Q', `${headSize}D`),
				terminal('k', 0, 132, 'K', `${headSize}D`),
				terminal('v', 0, 210, 'V', `${headSize}D`),
				node('query', 0.47, 54, 'RoPE · Q', `${headSize}D`),
				node('key', 0.47, 132, 'RoPE · K', `${headSize}D`),
				terminal('qo', 1, 54, 'Q′', `${headSize}D`),
				terminal('ko', 1, 132, 'K′', `${headSize}D`),
				terminal('vo', 1, 210, 'V · unchanged', `${headSize}D`)
			],
			[
				edge('q', 'query', 'query'),
				edge('query', 'qo', 'query'),
				edge('k', 'key', 'key'),
				edge('key', 'ko', 'key'),
				edge('v', 'vo', '')
			],
			258,
			'Q′ and K′ enter the dot product · V bypasses RoPE'
		);
	if (kind === 'residual')
		return graph(
			[
				terminal('x', 0, 120, 'Block X', `${hiddenSize}D`),
				node('attn', 0.24, 120, 'Attention', '', 'attention'),
				node('add1', 0.47, 120, '+ residual', `${hiddenSize}D`, 'residual'),
				node('ffn', 0.7, 120, 'Feed-forward', '', 'up'),
				node('add2', 0.96, 120, '+ residual', `${hiddenSize}D`, 'residual')
			],
			[
				edge('x', 'attn', 'attention'),
				edge('attn', 'add1', 'residual'),
				edge('x', 'add1', 'residual', 'X', 48),
				edge('add1', 'ffn', 'up'),
				edge('ffn', 'add2', 'residual'),
				edge('add1', 'add2', 'residual', 'After attention', 48)
			],
			172,
			'Each branch adds to its matching hidden state',
			0.16
		);
	if (kind === 'connector')
		return graph(
			[
				terminal('patches', 0, 89, 'Vision output', '256 × 768'),
				node('pack', 0.3, 89, 'Pixel shuffle', '16 × 12288', 'connector'),
				node('project', 0.64, 89, 'Wconnector', '12288 → 576', 'connector'),
				terminal('out', 1, 89, 'Visual tokens', '16 × 576')
			],
			[
				edge('patches', 'pack', 'connector'),
				edge('pack', 'project', 'connector'),
				edge('project', 'out', 'connector')
			],
			146,
			'After the vision encoder → image positions in the language input'
		);
	if (kind === 'embedding')
		return graph(
			[
				terminal('x', 0, 85, 'Patches', `${hiddenSize}D`),
				node('projection', 0.32, 85, 'Wpatch', `${hiddenSize} → ${outputSize}`, 'embedding'),
				node('position', 0.69, 85, '+ position', `${outputSize}D`, 'embedding'),
				terminal('out', 1, 85, 'Tokens', `${outputSize}D`)
			],
			[
				edge('x', 'projection', 'embedding'),
				edge('projection', 'position', 'embedding'),
				edge('position', 'out', 'embedding')
			],
			142,
			'Patch projection + learned positional embeddings'
		);
	if (kind === 'language-embedding')
		return graph(
			[
				terminal('text', 0, 53, 'Text IDs', 'lookup'),
				terminal('image', 0, 138, 'Visual tokens', '16 × 576'),
				node('embed', 0.38, 53, 'Embedding', `${hiddenSize}D`, 'embedding'),
				node('insert', 0.69, 96, 'Insert image tokens', `${hiddenSize}D`, 'embedding'),
				terminal('out', 1, 96, 'Language input', `${hiddenSize}D`)
			],
			[
				edge('text', 'embed', 'embedding'),
				edge('embed', 'insert', 'embedding'),
				edge('image', 'insert', 'connector'),
				edge('insert', 'out', 'embedding')
			],
			182,
			'Text and visual positions enter the same decoder'
		);
	const isNorm = kind === 'norm',
		isOutput = kind === 'classifier';
	const action = isNorm ? 'norm' : isOutput ? 'classifier' : 'projection';
	return graph(
		[
			terminal('x', 0, 82, isOutput ? 'Final hidden' : 'Token', `${hiddenSize}D`),
			node(
				'op',
				0.4,
				82,
				isNorm ? normalization : label,
				isNorm ? `${hiddenSize}D` : `${hiddenSize} → ${outputSize}`,
				action
			),
			...(isOutput ? [node('softmax', 0.74, 82, 'Softmax', `${outputSize} classes`, action)] : []),
			terminal(
				'out',
				1,
				82,
				isNorm ? 'Normalized X' : isOutput ? 'Probabilities' : 'Output',
				`${outputSize}D`
			)
		],
		[
			edge('x', 'op', action),
			...(isOutput
				? [edge('op', 'softmax', action), edge('softmax', 'out', action)]
				: [edge('op', 'out', action)])
		],
		143,
		isNorm
			? 'Normalize each token independently'
			: isOutput
				? 'After every model block · full vocabulary / class scores'
				: 'Recorded projection dimensions'
	);
}
