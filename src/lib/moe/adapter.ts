import { get } from 'svelte/store';
import {
	attentionHeadIdx, attentionHeadIdxTemp, blockIdx, expandedBlock,
	highlightedToken, hoveredMatrixCell, inputText, isFetchingModel, isLoaded,
	isModelRunning, isTextbookOpen, modelData, modelMetaMap, moeDocument, moeExample, moeModel,
	moeTokenIndex, predictedToken, sampling, selectedExampleIdx, selectedModel,
	temperature, tokenIds, tokens
} from '~/store';
import type { TraceDocument, TraceExample } from './types';

function tensor(data: number[][]): ModelData['outputs'][string] {
	return { data, dims: [data.length, data[0]?.length ?? 0], size: data.reduce((sum, row) => sum + row.length, 0) };
}

/** Keep vocabulary IDs separate from these whitespace-only display substitutions. */
function displayToken(text: string): string {
	return text.replace(/\n/g, '[NEWLINE]').replace(/\t/g, '[TAB]').replace(/\r/g, '[CR]')
		.replace(/\s{2,}/g, (spaces) => `[${spaces.length} SPACES]`);
}

/** Recompute a distribution from actual full-vocabulary logits, without model inference. */
export function recordedProbabilities(
	logits: number[], vocabulary: string[], selectedTemperature: number, selectedSampling: Sampling,
	visibleCandidates = 16
): Probabilities {
	if (!logits.length || logits.length !== vocabulary.length || !logits.every(Number.isFinite)) {
		throw new Error('Recorded vocabulary logits are missing or do not match the tokenizer vocabulary.');
	}
	if (!(selectedTemperature > 0) || !Number.isFinite(selectedTemperature)) {
		throw new Error('Temperature must be a finite number above zero.');
	}
	const ranked = logits.map((logit, tokenId) => ({ logit, tokenId }))
		.sort((a, b) => b.logit - a.logit || a.tokenId - b.tokenId);
	const scaled = ranked.map(({ logit }) => logit / selectedTemperature);
	const exp = scaled.map((value) => Math.exp(value - scaled[0]));
	const allMass = exp.reduce((sum, value) => sum + value, 0);
	const fullProbabilities = exp.map((value) => value / allMass);
	const cumulative: number[] = [];
	let cumulativeMass = 0;
	for (const probability of fullProbabilities) {
		cumulativeMass += probability;
		cumulative.push(cumulativeMass);
	}

	let cutoffIndex: number;
	if (selectedSampling.type === 'top-k') {
		const k = Math.min(ranked.length, Math.max(1, Math.floor(selectedSampling.value)));
		if (!Number.isFinite(k)) throw new Error('Top-k must be a finite count.');
		cutoffIndex = k - 1;
	} else {
		const p = Math.min(1, Math.max(0, selectedSampling.value));
		if (!Number.isFinite(p)) throw new Error('Top-p must be a finite probability.');
		// p=1 means no filtering, even if floating-point accumulation rounds to
		// one before the very small probabilities at the end of the vocabulary.
		cutoffIndex = p === 1 ? ranked.length - 1 : cumulative.findIndex((value) => value >= p);
		if (cutoffIndex < 0) cutoffIndex = ranked.length - 1;
	}
	const retainedMass = exp.slice(0, cutoffIndex + 1).reduce((sum, value) => sum + value, 0);
	return ranked.slice(0, visibleCandidates).map(({ logit, tokenId }, rank) => ({
		rank, tokenId, token: displayToken(vocabulary[tokenId]), logit,
		scaledLogit: scaled[rank], expLogit: exp[rank],
		probability: rank <= cutoffIndex ? exp[rank] / retainedMass : 0,
		topKLogit: rank <= cutoffIndex ? scaled[rank] : -Infinity,
		topPProbability: fullProbabilities[rank],
		cumulativeProbability: cumulative[rank], cutoffIndex
	}));
}

/** Adapt recorded tensors to the original components' names; never synthesize activations. */
export function toOriginalModelData(
	example: TraceExample, vocabulary: string[], selectedTemperature = 1,
	selectedSampling: Sampling = { type: 'top-p', value: 1 },
	attentionMultiplier = 1 / 64
): ModelData {
	if (!(attentionMultiplier > 0) || !Number.isFinite(attentionMultiplier)) {
		throw new Error('The recorded attention multiplier must be finite and positive.');
	}
	const outputs: ModelData['outputs'] = {};
	for (const [layer, heads] of example.attention.entries()) {
		for (const [head, probabilities] of heads.entries()) {
			const prefix = `block_${layer}_attn_head_${head}`;
			const scores = example.attentionScores[layer][head];
			const masked = example.attentionMaskedScores[layer][head].map((row, query) =>
				row.map((value, key) => key > query ? -Infinity : value));
			// The original first stage displays QKᵀ before scaling. These raw dot
			// products are derived by undoing the captured model's multiplier.
			// Granite's power-of-two factor makes this inverse exact before the
			// capture's decimal serialization; no Q/K activations are fabricated.
			outputs[`${prefix}_attn`] = tensor(scores.map((row) => row.map((value) => value / attentionMultiplier)));
			outputs[`${prefix}_attn_scaled`] = tensor(scores);
			outputs[`${prefix}_attn_masked`] = tensor(masked);
			outputs[`${prefix}_attn_softmax`] = tensor(probabilities);
			outputs[`${prefix}_attn_dropout`] = tensor(probabilities);
		}
	}
	const probabilities = recordedProbabilities(example.nextTokenLogits, vocabulary, selectedTemperature, selectedSampling);
	// Highlight the highest-probability candidate. A replay never invents a continuation.
	return { logits: example.nextTokenLogits, outputs, probabilities, sampled: probabilities[0] };
}

/** Bind a selected recording to the upstream layout and controls. */
export function hydrateRecordedExample(document: TraceDocument, exampleIndex = 0): void {
	const example = document.examples[exampleIndex];
	if (!example) throw new Error(`Recorded example ${exampleIndex} does not exist.`);
	const adapted = toOriginalModelData(example, document.vocabulary, get(temperature), get(sampling), document.model.attentionMultiplier);
	modelMetaMap['granite-moe'] = {
		layer_num: document.model.numLayers,
		attention_head_num: document.model.attentionHeads,
		dimension: document.model.hiddenSize
	};
	moeDocument.set(document);
	moeModel.set(document.model);
	moeExample.set(example);
	selectedModel.set('granite-moe');
	selectedExampleIdx.set(exampleIndex);
	inputText.set(example.prompt);
	tokens.set(example.tokens.map((token) => token.text));
	tokenIds.set(example.tokens.map((token) => token.id));
	moeTokenIndex.set(example.tokens.length - 1);
	modelData.set(adapted);
	predictedToken.set(adapted.sampled);
	expandedBlock.set({ id: null });
	isTextbookOpen.set(false);
	highlightedToken.set({ index: null, value: null, fix: false });
	hoveredMatrixCell.set({ row: null, col: null });
	blockIdx.set(0);
	attentionHeadIdx.set(0);
	attentionHeadIdxTemp.set(0);
	isFetchingModel.set(false);
	isModelRunning.set(false);
	isLoaded.set(true);
}

/** The original temperature and sampling controls operate on the captured vocabulary logits. */
export function applyRecordedSampling(): void {
	const document = get(moeDocument);
	const example = get(moeExample);
	if (!document || !example) return;
	const probabilities = recordedProbabilities(example.nextTokenLogits, document.vocabulary, get(temperature), get(sampling));
	modelData.update((current) => ({ ...current, probabilities, sampled: probabilities[0] }));
	predictedToken.set(probabilities[0]);
}
