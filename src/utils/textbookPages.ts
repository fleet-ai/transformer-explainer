import { get } from 'svelte/store';
import {
	expandedBlock,
	weightPopover,
	isBoundingBoxActive,
	textbookCurrentPageId,
	isExpandOrCollapseRunning,
	isFetchingModel,
	userId
} from '~/store';
import {
	highlightElements,
	removeHighlightFromElements,
	applyTransformerBoundingHeight,
	resetElementsHeight,
	highlightAttentionPath,
	removeAttentionPathHighlight,
	removeFingerFromElements
} from '~/utils/textbook';
import { drawResidualLine } from './animation';

export interface TextbookPage {
	id: string;
	title: string;
	content?: string;
	component?: any;
	timeoutId?: number;
	on: () => void;
	out: () => void;
	complete?: () => void;
}

const { drawLine, removeLine } = drawResidualLine();

export const textPages: TextbookPage[] = [
	{
		id: 'what-is-transformer',
		title: 'What is Transformer?',
		content: `<p><strong>Transformer</strong> is the core architecture behind modern AI, powering models like ChatGPT and Gemini. Introduced in 2017, it revolutionized how AI processes information. The same architecture is used for training on massive datasets and for inference to generate outputs. Here we replay recorded inference from IBM Granite 3.1 1B-A400M, a sparse mixture-of-experts transformer.</p>
`,
		on: () => {},
		out: () => {}
	},
	{
		id: 'how-transformers-work',
		title: 'How Transformers Work?',
		content: `<p>Transformers aren't magic—they build text step by step by asking:</p>
	<blockquote class="question">
		"What is the most probable next word that will follow this input?"
	</blockquote>
	<p>Here we explore how a trained model generates text. Write your own text or use an example, then click <strong>Generate</strong> to see it in action. If the model isn’t ready yet, try another <strong>Example</strong>.</p>`,
		on: () => {
			highlightElements(['.input-form']);
			if (get(isFetchingModel)) {
				highlightElements(['.input-form .select-button']);
			} else {
				highlightElements(['.input-form .generate-button']);
			}
		},
		out: () => {
			removeHighlightFromElements([
				'.input-form',
				'.input-form .select-button',
				'.input-form .generate-button'
			]);
		},
		complete: () => {
			removeFingerFromElements(['.input-form .select-button', '.input-form .generate-button']);
			if (get(textbookCurrentPageId) === 'how-transformers-work') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'how-transformers-work'
				});
			}
		}
	},
	{
		id: 'transformer-architecture',
		title: 'Transformer Architecture',
		content:
			'<p>Transformer has three main parts:</p><div class="numbered-list"><div class="numbered-item"><span class="number-circle">1</span><div class="item-content"><strong>Embeddings</strong> turn text into numbers.</div></div><div class="numbered-item"><span class="number-circle">2</span><div class="item-content"><strong>Transformer blocks</strong> mix information with Self-Attention and route each token through selected expert MLPs.</div></div><div class="numbered-item"><span class="number-circle">3</span><div class="item-content"><strong>Probabilities</strong> determine the likelihood of each next token.</div></div></div>',
		on: () => {
			const selectors = [
				'.step.embedding',
				'.step.softmax',
				'.transformer-bounding',
				'.transformer-bounding-title'
			];
			highlightElements(selectors);
			applyTransformerBoundingHeight(['.softmax-bounding', '.embedding-bounding']);
		},
		out: () => {
			const selectors = [
				'.step.embedding',
				'.step.softmax',
				'.transformer-bounding',
				'.transformer-bounding-title'
			];
			removeHighlightFromElements(selectors);
			resetElementsHeight(['.softmax-bounding', '.embedding-bounding']);
		}
	},
	{
		id: 'embedding',
		title: 'Embedding',
		content: `<p>Before a Transformer can use text, it first breaks it into small units and represents each as a list of numbers (vector). This process is called <strong>embedding</strong>, and the term can refer to both the process and the resulting vector.</p><p>In this tool, each vector appears as a rectangle, and hovering over it shows its size.</p>`,
		on: () => {
			highlightElements(['.step.embedding .title']);
		},
		out: () => {
			removeHighlightFromElements(['.step.embedding .title']);
		},
		complete: () => {
			removeFingerFromElements(['.step.embedding .title']);
			if (get(textbookCurrentPageId) === 'embedding') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'embedding'
				});
			}
		}
	},
	{
		id: 'token-embedding',
		title: 'Token Embedding',
		content: `<p><strong>Tokenization</strong> splits input text into tokens—small units like words or parts of words. Granite has a 49,152-token vocabulary, each with a unique ID.</p><p>In the <strong>token embedding</strong> step, every token is matched to a 1,024-number vector from a large lookup table. These vectors are learned during training to best represent each token’s meaning.</p>`,
		on: function () {
			const selectors = [
				'.token-column .column.token-string',
				'.token-column .column.token-embedding'
			];
			if (get(expandedBlock).id !== 'embedding') {
				expandedBlock.set({ id: 'embedding' });
				this.timeoutId = setTimeout(() => {
					highlightElements(selectors);
				}, 500);
			} else {
				highlightElements(selectors);
			}
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			const selectors = [
				'.token-column .column.token-string',
				'.token-column .column.token-embedding'
			];
			removeHighlightFromElements(selectors);
			if (get(textbookCurrentPageId) !== 'positional-encoding') expandedBlock.set({ id: null });
		}
	},
	{
		id: 'positional-encoding',
		title: 'Positional Encoding',
		content: `<p>Word order matters in language. <strong>Positional encoding</strong> gives each token information about its place in the sequence.</p><p>Granite uses RoPE, which rotates query and key vectors inside attention. Its token embeddings are multiplied by 12; there is no learned positional embedding added to them.</p>`,
		on: function () {
			const selectors = [
				'.token-column .column.position-embedding',
				'.token-column .column.symbol'
			];
			if (get(expandedBlock).id !== 'embedding') {
				expandedBlock.set({ id: 'embedding' });
				this.timeoutId = setTimeout(() => {
					highlightElements(selectors);
				}, 500);
			} else {
				highlightElements(selectors);
			}
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			const selectors = [
				'.token-column .column.position-embedding',
				'.token-column .column.symbol'
			];
			removeHighlightFromElements(selectors);
			if (get(textbookCurrentPageId) !== 'token-embedding') expandedBlock.set({ id: null });
		}
	},
	{
		id: 'blocks',
		title: 'Repetitive Transformer Blocks',
		content: `<p>A <strong>Transformer block</strong> is the main unit of processing in the model. It has two parts:</p><ul><li><strong>Multi-head self-attention</strong> – lets tokens share information</li><li><strong>Mixture of experts</strong> – routes each token through eight of 32 expert networks</li></ul><p>Models stack many blocks so token representations become richer as they pass through. This Granite model has 24 of them.</p>`,
		on: function () {
			this.timeoutId = setTimeout(
				() => {
					highlightElements([
						'.transformer-bounding',
						'.step.transformer-blocks .guide',
						'.attention > .title',
						'.mlp > .title'
					]);
					highlightElements(['.transformer-bounding-title'], 'textbook-button-highlight');
					isBoundingBoxActive.set(true);
				},
				get(isExpandOrCollapseRunning) ? 500 : 0
			);
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			removeHighlightFromElements([
				'.transformer-bounding',
				'.step.transformer-blocks .guide',
				'.attention > .title',
				'.mlp > .title'
			]);
			removeHighlightFromElements(['.transformer-bounding-title'], 'textbook-button-highlight');
			isBoundingBoxActive.set(false);
		},
		complete: () => {
			removeFingerFromElements(['.transformer-bounding-title']);
			if (get(textbookCurrentPageId) === 'blocks') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'blocks'
				});
			}
		}
	},
	{
		id: 'self-attention',
		title: 'Multi-Head Self Attention',
		content:
			'<p><strong>Self-attention</strong> lets the model decide which parts of the input are most relevant to each token. This helps it capture meaning and relationships, even between far-apart words.</p><p>In <strong>multi-head</strong> form, the model runs several attention processes in parallel, each focusing on different patterns in the text.</p>',
		on: () => {
			highlightElements(['.step.attention']);
		},
		out: () => {
			removeHighlightFromElements(['.step.attention']);
		}
	},
	{
		id: 'qkv',
		title: 'Query, Key, Value',
		content: `<p>Bias-free projections transform each token’s RMS-normalized hidden state into <span class="blue">Q (1024)</span>, <span class="red">K (512)</span>, and <span class="green">V (512)</span>.</p><p>Q splits into <strong>16 heads of 64 coordinates</strong>; K and V each split into <strong>8 heads of 64</strong>. RoPE rotates coordinate pairs within Q and K using token position. It preserves the 64-dimensional head size and leaves V unchanged.</p><p>Each pair of query heads shares one K/V head: Q1 and Q2 use K1/V1, Q3 and Q4 use K2/V2, and so on. Click <strong>Inspect RoPE</strong> to follow the recorded rotation.</p>`,
		on: function () {
			this.timeoutId = setTimeout(
				() => {
					highlightElements(['g.path-group.qkv', '.step.qkv .qkv-column']);
				},
				get(isExpandOrCollapseRunning) ? 500 : 0
			);
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			removeHighlightFromElements(['g.path-group.qkv', '.step.qkv .qkv-column']);
			weightPopover.set(null);
		},
		complete: () => {
			removeFingerFromElements(['.step.qkv .qkv-column']);
			if (get(textbookCurrentPageId) === 'qkv') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'qkv'
				});
			}
		}
	},

	{
		id: 'multi-head',
		title: 'Multi-head',
		content:
			'<p>Granite uses <strong>grouped-query attention</strong>: 16 query heads share 8 key/value heads. All heads contain 64 coordinates; Q is larger overall because it has twice as many heads.</p><p>After RoPE, query heads 1 and 2 both compare against key head 1, then each uses its own attention weights to combine value head 1. The same sharing applies to the other query pairs. K/V sharing repeats the same values without creating new projections.</p><p>Use the head arrows to follow the highlighted query head and its shared K/V head in the diagram.</p>',
		on: () => {
			highlightAttentionPath();
			highlightElements(['.multi-head .head-title']);
		},
		out: () => {
			removeAttentionPathHighlight();
			removeHighlightFromElements(['.multi-head .head-title']);
		},
		complete: () => {
			removeFingerFromElements(['.multi-head .head-title']);
			if (get(textbookCurrentPageId) === 'multi-head') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'multi-head'
				});
			}
		}
	},
	{
		id: 'masked-self-attention',
		title: 'Masked Self Attention',
		content: `<p>In each head, the model decides how much each token focuses on others:</p><ul><li><strong>Dot Product</strong> – Multiply matching numbers in <span class="blue">Query</span>/<span class="red">Key</span> vectors, sum to get <span class="purple">attention scores</span>.</li><li><strong>Mask</strong> – Hide future tokens so it can't peek ahead.</li><li><strong>Softmax</strong> – Convert scores to probabilities, each row summing to 1, showing focus on earlier tokens.</li></ul>`,
		on: () => {
			highlightAttentionPath();
			highlightElements(['.attention-matrix.attention-result']);
		},
		out: () => {
			removeAttentionPathHighlight();
			removeHighlightFromElements(['.attention-matrix.attention-result']);
			expandedBlock.set({ id: null });
		},
		complete: () => {
			removeFingerFromElements(['.attention-matrix.attention-result']);
			if (get(textbookCurrentPageId) === 'masked-self-attention') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'masked-self-attention'
				});
			}
		}
	},
	{
		id: 'output-concatenation',
		title: 'Attention Output & Concatenation',
		content:
			'<p>Each head <span class="highlight">multiplies its <span class="purple">attention scores</span> with the <span class="green">Value</span> embeddings to produce its attention output</span>—a refined representation of each token after considering context.</p><p>Granite has 16 query-head outputs, which are concatenated and projected to a 1,024-number vector. Each pair of query heads shares a key/value head.</p>',
		on: function () {
			this.timeoutId = setTimeout(
				() => {
					highlightElements(['path.to-attention-out.value-to-out', '.attention .column.out']);
				},
				get(isExpandOrCollapseRunning) ? 500 : 0
			);
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			removeHighlightFromElements(['path.to-attention-out.value-to-out', '.attention .column.out']);
			weightPopover.set(null);
		},
		complete: () => {
			removeFingerFromElements(['.attention .column.out']);
			if (get(textbookCurrentPageId) === 'output-concatenation') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'output-concatenation'
				});
			}
		}
	},
	{
		id: 'mlp',
		title: 'MoE (Mixture of Experts)',
		content:
			'<p>A learned <strong>router</strong> scores 32 experts for each token. Granite selects the top eight logits and applies softmax to those eight scores.</p><p>Each selected expert is a SwiGLU network. Its output is multiplied by its routing weight, then the eight outputs are summed. The recorded gates and outputs are visible in the MoE panel.</p>',
		on: () => {
			highlightElements(['.step.mlp', '.operation-col.activation']);
		},
		out: () => {
			removeHighlightFromElements(['.step.mlp', '.operation-col.activation']);
		}
	},

	{
		id: 'output-logit',
		title: 'Output Logit',
		content: `<p>After all Transformer blocks, the last token's output embedding, enriched with context from all previous tokens, is multiplied by learned weights in a final layer.</p><p>This produces <strong>logits</strong>, 49,152 numbers—one for each token in Granite’s vocabulary—that indicate how likely each token is to come next.</p>`,
		on: () => {
			highlightElements(['g.path-group.softmax', '.column.final']);
		},
		out: () => {
			removeHighlightFromElements(['g.path-group.softmax', '.column.final']);
			weightPopover.set(null);
		},
		complete: () => {
			removeFingerFromElements(['.column.final']);
			if (get(textbookCurrentPageId) === 'output-logit') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'output-logit'
				});
			}
		}
	},
	{
		id: 'output-probabilities',
		title: 'Probabilities',
		content:
			'<p>Logits are just raw scores. To make them easier to interpret, we convert them into <strong>probabilities</strong> between 0 and 1, where all add up to 1. This tells us the likelihood of each token being the next word.</p><p>Instead of always picking the highest-probability token, we can use different selection strategies to balance safety and creativity in the generated text.</p>',
		on: () => {
			highlightElements(['.step.softmax .title']);
		},
		out: () => {
			removeHighlightFromElements(['.step.softmax .title']);
		},
		complete: () => {
			removeFingerFromElements(['.step.softmax .title']);
			if (get(textbookCurrentPageId) === 'output-probabilities') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'output-probabilities'
				});
			}
		}
	},
	{
		id: 'temperature',
		title: 'Temperature',
		content:
			'<p><strong>Temperature</strong> works by scaling the logits before turning them into probabilities. A <strong>low temperature</strong> (e.g., 0.2) makes large logits even larger and small ones smaller, favoring the highest-scoring tokens and leading to more <strong>predictable choices</strong>. A <strong>high temperature</strong> (e.g., 1.0 or above) flattens the differences, making less likely tokens more competitive and leading to more <strong>creative outputs</strong>.</p>',
		on: function () {
			if (get(expandedBlock).id !== 'softmax') {
				expandedBlock.set({ id: 'softmax' });
				this.timeoutId = setTimeout(() => {
					highlightElements([
						'.formula-step.scaled',
						'.title-box.scaled',
						'.content-box.scaled',
						'.temperature-input'
					]);
				}, 500);
			} else {
				highlightElements([
					'.formula-step.scaled',
					'.title-box.scaled',
					'.content-box.scaled',
					'.temperature-input'
				]);
			}
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			removeHighlightFromElements([
				'.formula-step.scaled',
				'.title-box.scaled',
				'.temperature-input',
				'.content-box.scaled'
			]);
			if (!['temperature', 'sampling'].includes(get(textbookCurrentPageId)))
				expandedBlock.set({ id: null });
		},
		complete: () => {
			removeFingerFromElements(['.temperature-input']);
			if (get(textbookCurrentPageId) === 'temperature') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'temperature'
				});
			}
		}
	},
	{
		id: 'sampling',
		title: 'Sampling Strategy',
		content:
			'<p>Finally, we need a strategy to pick the next token. Many exist, but here are common ones: Greedy search picks the top one. <strong>Top-k</strong> keeps only the k most likely tokens, and <strong>top-p</strong> keeps the smallest set whose total probability is at least p—trimming unlikely ones early.</p><p>Then softmax turns the remaining logits into probabilities, and one token is picked at random from the allowed set.</p>',
		on: function () {
			if (get(expandedBlock).id !== 'softmax') {
				expandedBlock.set({ id: 'softmax' });
				this.timeoutId = setTimeout(() => {
					highlightElements([
						'.formula-step.sampling',
						'.title-box.sampling',
						'.sampling-input',
						'.content-box.sampling'
					]);
				}, 500);
			} else {
				highlightElements([
					'.formula-step.sampling',
					'.title-box.sampling',
					'.sampling-input',
					'.content-box.sampling'
				]);
			}
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			removeHighlightFromElements([
				'.formula-step.sampling',
				'.title-box.sampling',
				'.sampling-input',
				'.content-box.sampling'
			]);
			if (!['temperature', 'sampling'].includes(get(textbookCurrentPageId)))
				expandedBlock.set({ id: null });
		},
		complete: () => {
			removeFingerFromElements(['.sampling-input']);
			if (get(textbookCurrentPageId) === 'sampling') {
				window.dataLayer?.push({
					user_id: get(userId),
					event: `textbook-complete`,
					page_id: 'sampling'
				});
			}
		}
	},
	{
		id: 'residual',
		title: 'Residual Connection',
		content: `<p>Transformers have auxiliary features that enhance the model performance. For example, a <strong>residual connection</strong> adds a layer's input to its output, keeping information from fading through many blocks. Granite uses two residual additions per block; each attention or MoE output is multiplied by 0.22 before being added to the residual stream.</p>`,
		on: function () {
			this.timeoutId = setTimeout(
				() => {
					highlightElements(['.operation-col.residual', '.residual-start']);
					drawLine();
				},
				get(isExpandOrCollapseRunning) ? 500 : 0
			);
		},
		out: function () {
			if (this.timeoutId) {
				clearTimeout(this.timeoutId);
				this.timeoutId = undefined;
			}
			removeHighlightFromElements(['.operation-col.residual', '.residual-start']);
			removeLine();
		}
	},
	{
		id: 'layer-normalization',
		title: 'Layer Normalization',
		content: `<p><strong>RMSNorm</strong> rescales a vector by its root-mean-square magnitude, followed by learned coordinate weights. Granite applies it before self-attention, before the MoE, and once more before the final output.</p>`,
		on: () => {
			highlightElements(['.operation-col.ln']);
		},
		out: () => {
			removeHighlightFromElements(['.operation-col.ln']);
		}
	},
	{
		id: 'dropout',
		title: 'Dropout',
		content: `<p>During training, <strong>dropout</strong> randomly turns off some connections between numbers so the model doesn't overfit to specific patterns. This helps it learn features that generalize better. This recording runs in evaluation mode. In inference, dropout is turned off.</p>`,
		on: () => {
			highlightElements(['.operation-col.dropout']);
		},
		out: () => {
			removeHighlightFromElements(['.operation-col.dropout']);
		}
	}
	// {
	// 	id: 'final',
	// 	title: `Let's explore!`,
	// 	content: '',
	// 	on: () => {},
	// 	out: () => {}
	// }
];
