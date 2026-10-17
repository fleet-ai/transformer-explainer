import type { ArchitectureId, RecordedLayer, RecordedOperation, RecordedTensor } from './types';

export interface Stage {
	id: string;
	label: string;
	note: string;
	x: number;
	y: number;
	w: number;
	h: number;
	color: 'blue' | 'purple' | 'green' | 'pink';
	aliases?: string[];
	visual?: 'tokens' | 'patches' | 'output';
}
interface Edge {
	from: string;
	to: string;
	label?: string;
	color?: string;
	path?: string;
	labelPosition?: { x: number; y: number };
}
export interface Diagram {
	title: string;
	caption: string;
	stages: Stage[];
	edges: Edge[];
	lanes?: { y: number; title: string }[];
}

const stage = (
	id: string,
	label: string,
	note: string,
	x: number,
	y: number,
	color: Stage['color'] = 'purple',
	aliases: string[] = [],
	w = 190,
	h = 136
): Stage => ({ id, label, note, x, y, w, h, color, aliases });
const input = (x: number, y: number, image = false): Stage => ({
	...stage(
		'input',
		image ? 'Image patches' : 'Input tokens',
		image ? 'Select a patch' : 'Select a token',
		x,
		y,
		'blue',
		[],
		150,
		250
	),
	visual: image ? 'patches' : 'tokens'
});
const output = (x: number, y: number): Stage => ({
	...stage(
		'output',
		'Output',
		'Recorded model output',
		x,
		y,
		'purple',
		['logits', 'classifier', 'lm-head', 'prediction'],
		194,
		250
	),
	visual: 'output'
});

export const diagrams: Record<ArchitectureId, Diagram> = {
	mla: {
		title: 'Multi-head Latent Attention',
		caption:
			'A compact latent vector carries K/V content. Position is carried separately by the rotary key.',
		stages: [
			input(28, 252),
			stage(
				'embedding',
				'Hidden state',
				'Input to the recorded attention layer',
				222,
				309,
				'blue',
				['input']
			),
			stage('compress', 'Compress K/V', 'Down projection → latent cache', 474, 174, 'purple', [
				'kv-compress',
				'kv-down',
				'latent'
			]),
			stage('query', 'Query projection', 'Queries for each attention head', 474, 446, 'blue', [
				'q-projection',
				'query-projection'
			]),
			stage('expand', 'Expand K/V', 'Recover per-head content', 746, 174, 'green', [
				'kv-expand',
				'kv-up'
			]),
			stage('rope', 'Decoupled RoPE', 'Rotate positional Q and shared K', 746, 446, 'pink', [
				'rotary'
			]),
			stage('attention', 'Attention', 'Content + position scores → × V', 1020, 309, 'purple', [
				'mla-attention',
				'self-attention'
			]),
			output(1286, 252)
		],
		edges: [
			{ from: 'input', to: 'embedding' },
			{ from: 'embedding', to: 'compress' },
			{ from: 'embedding', to: 'query' },
			{ from: 'compress', to: 'expand', label: 'latent cᴷⱽ' },
			{ from: 'compress', to: 'rope', label: 'shared positional K' },
			{ from: 'query', to: 'rope', label: '64D rotary Q' },
			{
				from: 'query',
				to: 'attention',
				label: '128D content Q',
				path: 'M 664 514 C 688 514, 688 400, 712 400 L 970 400 C 996 400, 996 377, 1020 377',
				labelPosition: { x: 842, y: 386 }
			},
			{ from: 'expand', to: 'attention' },
			{ from: 'rope', to: 'attention' },
			{ from: 'attention', to: 'output' }
		]
	},
	'encoder-decoder': {
		title: 'Encoder–decoder Transformer',
		caption:
			'The encoder reads the whole source. The decoder attends to its past and to the encoder’s memory.',
		stages: [
			input(28, 174),
			stage('embedding', 'Source embedding', 'Token IDs → hidden vectors', 224, 206, 'blue'),
			stage(
				'encoder-self',
				'Encoder self-attention',
				'Bidirectional · all source tokens',
				474,
				206,
				'purple',
				['encoder-attention']
			),
			stage(
				'memory',
				'Encoder memory',
				'Final encoder output · keys and values',
				1004,
				206,
				'green',
				['encoder-output', 'encoder-memory']
			),
			stage(
				'decoder-input',
				'Decoder input',
				'Start token + recorded target prefix',
				224,
				516,
				'blue',
				['decoder-embedding']
			),
			stage(
				'decoder-self',
				'Decoder self-attention',
				'Causal · no future target tokens',
				474,
				516,
				'purple',
				['decoder-attention']
			),
			stage(
				'cross-attention',
				'Cross-attention',
				'Target queries × source keys',
				724,
				516,
				'pink',
				['cross']
			),
			stage(
				'feed-forward',
				'Decoder feed-forward',
				'Transform each target position',
				1004,
				516,
				'green',
				['decoder-ffn', 'ffn']
			),
			output(1286, 452)
		],
		edges: [
			{ from: 'input', to: 'embedding' },
			{ from: 'embedding', to: 'encoder-self' },
			{ from: 'encoder-self', to: 'memory', label: 'remaining encoder blocks' },
			{ from: 'decoder-input', to: 'decoder-self' },
			{ from: 'decoder-self', to: 'cross-attention' },
			{ from: 'memory', to: 'cross-attention', label: 'K/V: source' },
			{ from: 'cross-attention', to: 'feed-forward' },
			{ from: 'feed-forward', to: 'output' }
		],
		lanes: [
			{ y: 155, title: 'ENCODER · read the source' },
			{ y: 465, title: 'DECODER · predict a target token' }
		]
	},
	hybrid: {
		title: 'Transformer–Mamba Hybrid',
		caption:
			'A recurrent state carries history through Mamba blocks; attention also reads individual token positions.',
		stages: [
			input(28, 252),
			stage(
				'embedding',
				'Residual stream',
				'Current hidden state and original embedding',
				222,
				309,
				'blue',
				['input']
			),
			stage(
				'attention',
				'Shared transformer',
				'Concatenated [hidden; embedding] · 4096D',
				474,
				174,
				'pink',
				['self-attention', 'hybrid-attention']
			),
			stage(
				'feed-forward',
				'Gated feed-forward',
				'Transformer output joins residual stream',
				746,
				174,
				'purple',
				['ffn']
			),
			stage('conv', 'Local convolution', 'Mamba · causal convolution + SiLU', 746, 446, 'green', [
				'convolution',
				'mamba-conv'
			]),
			stage(
				'selective-state',
				'Selective state',
				'Input-dependent recurrent update',
				1020,
				446,
				'purple',
				['state-space', 'mamba', 'ssm', 'recurrence']
			),
			output(1286, 389)
		],
		edges: [
			{ from: 'input', to: 'embedding' },
			{ from: 'embedding', to: 'attention' },
			{ from: 'attention', to: 'feed-forward' },
			{ from: 'feed-forward', to: 'conv' },
			{ from: 'embedding', to: 'conv', label: 'residual addition' },
			{ from: 'conv', to: 'selective-state' },
			{ from: 'selective-state', to: 'output' }
		]
	},
	'vision-language': {
		title: 'Vision–language Transformer',
		caption:
			'Image patches become visual tokens. A connector places them in the language model’s hidden space.',
		stages: [
			input(28, 154, true),
			stage('patches', 'Patch embedding', 'Image regions → visual vectors', 224, 206, 'blue', [
				'patch-embedding'
			]),
			stage(
				'vision-attention',
				'Vision encoder',
				'Bidirectional patch attention',
				474,
				206,
				'green',
				['vision', 'image-attention']
			),
			stage(
				'projector',
				'Vision → language',
				'Connector / visual token projection',
				724,
				206,
				'pink',
				['connector', 'projection']
			),
			stage('text-input', 'Text prompt', 'Token IDs → learned text embeddings', 224, 516, 'blue'),
			stage(
				'embedding',
				'Merge image + text',
				'Visual vectors replace image placeholders',
				724,
				516,
				'blue',
				['text-embedding']
			),
			stage(
				'attention',
				'Language decoder',
				'Causal attention over image + text',
				1004,
				362,
				'purple',
				['text-attention', 'decoder-attention']
			),
			output(1286, 306)
		],
		edges: [
			{ from: 'input', to: 'patches' },
			{ from: 'patches', to: 'vision-attention' },
			{ from: 'vision-attention', to: 'projector' },
			{ from: 'projector', to: 'embedding', label: 'visual tokens' },
			{ from: 'text-input', to: 'embedding', label: 'text embeddings' },
			{ from: 'embedding', to: 'attention', label: 'one sequence' },
			{ from: 'attention', to: 'output' }
		],
		lanes: [
			{ y: 155, title: 'VISION · encode the image' },
			{ y: 465, title: 'LANGUAGE · read the prompt' }
		]
	},
	vit: {
		title: 'Vision Transformer',
		caption:
			'An image is a sequence of patches. The class token gathers evidence through bidirectional attention.',
		stages: [
			input(28, 252, true),
			stage('patches', 'Patch embedding', 'Flatten patches → linear projection', 224, 309, 'blue', [
				'patch-embedding'
			]),
			stage(
				'embedding',
				'+ CLS + position',
				'A class token joins the patch sequence',
				470,
				309,
				'pink',
				['position', 'position-embedding']
			),
			stage(
				'attention',
				'Self-attention',
				'Bidirectional · class token + patches',
				716,
				309,
				'purple',
				['self-attention']
			),
			stage('pool', 'Class token', 'Final CLS representation', 1004, 309, 'green', [
				'cls',
				'pooling'
			]),
			output(1286, 252)
		],
		edges: [
			{ from: 'input', to: 'patches' },
			{ from: 'patches', to: 'embedding' },
			{ from: 'embedding', to: 'attention' },
			{ from: 'attention', to: 'pool', label: 'encoder blocks' },
			{ from: 'pool', to: 'output' }
		]
	}
};

export function diagramFor(architecture: ArchitectureId, layer?: RecordedLayer): Diagram {
	if (
		architecture !== 'hybrid' ||
		!layer ||
		layer.operations.some((operation) => operation.id === 'attention')
	)
		return diagrams[architecture];
	return {
		...diagrams.hybrid,
		stages: [
			input(28, 252),
			stage('embedding', 'Residual stream', 'Input to this Mamba block', 222, 309, 'blue', [
				'input'
			]),
			stage('conv', 'Local convolution', 'Causal convolution + SiLU', 474, 309, 'green'),
			stage('selective-state', 'Selective state', 'Input-dependent recurrent update', 746, 309),
			output(1286, 252)
		],
		edges: [
			{ from: 'input', to: 'embedding' },
			{ from: 'embedding', to: 'conv' },
			{ from: 'conv', to: 'selective-state' },
			{ from: 'selective-state', to: 'output', label: 'gated norm → output projection + residual' }
		]
	};
}

export function operationFor(stage: Stage, layer?: RecordedLayer): RecordedOperation | undefined {
	const ids = [stage.id, ...(stage.aliases ?? [])];
	return (
		layer?.operations.find((op) => ids.includes(op.id)) ??
		layer?.operations.find((op) => ids.includes(op.kind)) ??
		layer?.operations.find((op) =>
			ids.some((id) => op.id.endsWith(`.${id}`) || op.id.endsWith(`/${id}`))
		)
	);
}

export function previewTensor(operation?: RecordedOperation): RecordedTensor | undefined {
	return (
		operation?.tensors.find(
			(tensor) =>
				tensor.role === 'attention' || /attention.*(prob|weight)|probabilit/i.test(tensor.name)
		) ??
		operation?.tensors.find((tensor) => Array.isArray(tensor.values[0])) ??
		operation?.tensors[0]
	);
}

export function edgePath(from: Stage, to: Stage): string {
	if (to.y > from.y && to.x < from.x) {
		const x1 = from.x + from.w / 2,
			y1 = from.y + from.h;
		const x2 = to.x + to.w / 2,
			y2 = to.y;
		return `M ${x1} ${y1} C ${x1} ${y1 + 70}, ${x2} ${y2 - 70}, ${x2} ${y2}`;
	}
	if (from.x === to.x) {
		const x = from.x + from.w / 2;
		return `M ${x} ${from.y + from.h} L ${x} ${to.y}`;
	}
	const x1 = from.x + from.w,
		y1 = from.y + from.h / 2;
	const x2 = to.x,
		y2 = to.y + to.h / 2;
	const bend = Math.max(34, (x2 - x1) / 2);
	return `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`;
}
