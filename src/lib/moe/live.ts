import type { TraceDocument } from './types';

export const LOCAL_INFERENCE_ORIGIN = 'http://127.0.0.1:8766';
export const MAX_PROMPT_CHARACTERS = 4096;
export const MAX_PROMPT_TOKENS = 32;
const GRANITE_MODEL = 'ibm-granite/granite-3.1-1b-a400m-base';
const GRANITE_REVISION = '408b6e90baab8cf24f4aa9f8e19703ffa0a53b29';
const JOB_TIMEOUT_MS = 180_000;
const liveAssets = new Map<string, Record<TraceAsset, string>>();
export type TraceAsset = 'main' | 'attention' | 'expert';
export type LiveProgress = 'connecting' | 'loading' | 'queued' | 'running' | 'downloading';

export function promptError(prompt: string): string {
	if (!prompt.trim()) return 'Enter a prompt before generating.';
	if (prompt.length > MAX_PROMPT_CHARACTERS)
		return `Use at most ${MAX_PROMPT_CHARACTERS.toLocaleString()} characters (and ${MAX_PROMPT_TOKENS} tokens).`;
	return '';
}

function validId(id: unknown): id is string {
	return typeof id === 'string' && /^live-[A-Za-z0-9_-]{1,123}$/.test(id);
}

/** Live paths are bound to this loopback service, never to server-supplied origins. */
function localPath(value: unknown, expectedPath: string): string {
	if (value !== expectedPath)
		throw new Error('The local service returned an unexpected trace URL.');
	return `${LOCAL_INFERENCE_ORIGIN}${expectedPath}`;
}

export function isLiveTrace(exampleId: string): boolean {
	return validId(exampleId);
}

/** Bound full live tensors while keeping the small fixed preset catalog cached. */
export function trimLiveRecordings<T>(recordings: Map<string, T>, limit = 3): void {
	const liveIds = [...recordings.keys()].filter(isLiveTrace);
	for (const id of liveIds.slice(0, Math.max(0, liveIds.length - limit))) recordings.delete(id);
}

/** Shared by downloads and both lazy arithmetic loaders. */
export function traceAssetUrl(exampleId: string, kind: TraceAsset, base = ''): string {
	const registered = liveAssets.get(exampleId);
	if (registered) return registered[kind];
	if (isLiveTrace(exampleId))
		throw new Error('This live trace is no longer connected. Generate the prompt again.');
	return kind === 'main'
		? `${base}/traces/granite-moe.json`
		: `${base}/traces/${kind}-operations-${encodeURIComponent(exampleId)}.json`;
}

function abortError(): DOMException {
	return new DOMException('The request was canceled.', 'AbortError');
}

async function requestJson(
	url: string,
	signal: AbortSignal,
	timeout: number,
	init: RequestInit = {}
): Promise<any> {
	if (signal.aborted) throw abortError();
	const controller = new AbortController();
	const cancel = () => controller.abort();
	signal.addEventListener('abort', cancel, { once: true });
	const timer = setTimeout(cancel, timeout);
	try {
		const response = await fetch(url, {
			...init,
			signal: controller.signal,
			credentials: 'omit',
			cache: 'no-store',
			headers: { ...init.headers, ...(init.method === 'POST' ? { 'X-MoE-Client': 'explainer' } : {}) }
		});
		let data: any;
		try {
			data = await response.json();
		} catch {
			throw new Error(
				`The local service returned an unreadable response (HTTP ${response.status}).`
			);
		}
		if (!response.ok)
			throw new Error(
				typeof data?.error === 'string'
					? data.error
					: `The local service returned HTTP ${response.status}.`
			);
		return data;
	} catch (error) {
		if (signal.aborted) throw abortError();
		if (controller.signal.aborted)
			throw new Error(
				'Could not connect to Local Granite in time. Allow local-network access if prompted, or open the local app.'
			);
		if (error instanceof TypeError)
			throw new Error(
				'Cannot reach Local Granite. Start the local inference service on this Mac, allow local-network access if asked, then retry.'
			);
		throw error;
	} finally {
		clearTimeout(timer);
		signal.removeEventListener('abort', cancel);
	}
}

function pause(signal: AbortSignal): Promise<void> {
	return new Promise((resolve, reject) => {
		if (signal.aborted) {
			reject(abortError());
			return;
		}
		const cancel = () => {
			clearTimeout(timer);
			reject(abortError());
		};
		const timer = setTimeout(() => {
			signal.removeEventListener('abort', cancel);
			resolve();
		}, 1000);
		signal.addEventListener('abort', cancel, { once: true });
	});
}

function vector(value: unknown, length: number): value is number[] {
	return Array.isArray(value) && value.length === length && value.every(Number.isFinite);
}

function rows(value: unknown, count: number, width: number): value is number[][] {
	return Array.isArray(value) && value.length === count && value.every((row) => vector(row, width));
}

/** Reject incomplete or mismatched captures before touching the displayed diagram. */
export function validateLiveTrace(value: unknown, id: string, prompt: string): TraceDocument {
	const data = value as TraceDocument;
	if (
		data?.schemaVersion !== 1 ||
		data.model?.id !== GRANITE_MODEL ||
		data.model.revision !== GRANITE_REVISION ||
		data.model.numLayers !== 24 ||
		data.model.attentionHeads !== 16 ||
		data.model.keyValueHeads !== 8 ||
		data.model.attentionHeadSize !== 64 ||
		data.model.hiddenSize !== 1024 ||
		data.model.numExperts !== 32 ||
		data.model.topK !== 8 ||
		data.model.attentionMultiplier !== 1 / 64 ||
		!Array.isArray(data.examples) ||
		data.examples.length !== 1 ||
		data.examples[0]?.id !== id ||
		data.examples[0]?.prompt !== prompt ||
		!validId(id)
	) {
		throw new Error('The generated trace does not match this prompt and Granite model.');
	}
	const example = data.examples[0];
	const count = example.tokens?.length;
	const vocabularySize = data.model.vocabSize;
	if (
		!count ||
		count > MAX_PROMPT_TOKENS ||
		!Number.isInteger(vocabularySize) ||
		vocabularySize < 1 ||
		!Array.isArray(data.vocabulary) ||
		data.vocabulary.length !== vocabularySize ||
		!data.vocabulary.every((token) => typeof token === 'string') ||
		!vector(example.nextTokenLogits, vocabularySize) ||
		!example.tokens.every(
			(token, index) =>
				token.index === index &&
				Number.isInteger(token.id) &&
				token.id >= 0 &&
				token.id < vocabularySize &&
				typeof token.text === 'string'
		) ||
		!data.capture?.timestamp ||
		!data.capture.device ||
		!data.capture.dtype
	) {
		throw new Error('The generated trace has missing token, vocabulary, or capture data.');
	}
	for (const [preview, norms] of [
		[example.embeddingPreview, example.embeddingNorms],
		[example.finalHiddenPreview, example.finalHiddenNorms]
	] as const) {
		if (!rows(preview, count, 8) || !vector(norms, count))
			throw new Error('The generated trace has incomplete embedding values.');
	}
	for (const field of ['attention', 'attentionScores', 'attentionMaskedScores'] as const) {
		if (
			!Array.isArray(example[field]) ||
			example[field].length !== 24 ||
			!example[field].every(
				(heads) =>
					Array.isArray(heads) &&
					heads.length === 16 &&
					heads.every((head) => rows(head, count, count))
			)
		) {
			throw new Error('The generated trace has incomplete attention values.');
		}
	}
	if (!Array.isArray(example.layers) || example.layers.length !== 24)
		throw new Error('The generated trace has missing transformer blocks.');
	for (const [index, layer] of example.layers.entries()) {
		if (
			layer.index !== index ||
			!Array.isArray(layer.tokens) ||
			layer.tokens.length !== count ||
			!vector(layer.expertTokenCounts, 32) ||
			!rows(layer.attentionInputPreview, count, 8) ||
			!rows(layer.attentionOutputPreview, count, 8) ||
			!vector(layer.attentionInputNorms, count) ||
			!vector(layer.attentionOutputNorms, count)
		) {
			throw new Error('The generated trace has incomplete block values.');
		}
		for (const [preview, norms] of [
			[layer.queryPreview, layer.queryNorms],
			[layer.keyPreview, layer.keyNorms],
			[layer.valuePreview, layer.valueNorms]
		] as const) {
			if (
				!Array.isArray(preview) ||
				preview.length !== 16 ||
				!preview.every((head) => rows(head, count, 8)) ||
				!rows(norms, 16, count)
			) {
				throw new Error('The generated trace has incomplete Q/K/V values.');
			}
		}
		for (const token of layer.tokens) {
			if (
				!vector(token.routerLogits, 32) ||
				!vector(token.routerProbabilities, 32) ||
				!vector(token.routingWeights, 8) ||
				!vector(token.selectedExperts, 8) ||
				new Set(token.selectedExperts).size !== 8 ||
				!token.selectedExperts.every(
					(expert) => Number.isInteger(expert) && expert >= 0 && expert < 32
				) ||
				!vector(token.expertOutputNorms, 8) ||
				!vector(token.weightedOutputNorms, 8) ||
				!rows(token.expertOutputPreview, 8, 8) ||
				!vector(token.combinedOutputPreview, 8) ||
				![token.inputNorm, token.outputNorm, token.reconstructionMaxAbsError].every(Number.isFinite)
			) {
				throw new Error('The generated trace has incomplete expert routing values.');
			}
		}
	}
	return data;
}

/** A fresh forward pass produces one prompt, its next-token logits, and both operation sidecars. */
export async function generateLiveTrace(
	prompt: string,
	options: {
		signal: AbortSignal;
		onProgress?: (progress: LiveProgress) => void;
	}
): Promise<TraceDocument> {
	const invalid = promptError(prompt);
	if (invalid) throw new Error(invalid);
	const { signal, onProgress = () => {} } = options;
	onProgress('connecting');
	const loadingDeadline = Date.now() + JOB_TIMEOUT_MS;
	for (;;) {
		// Public HTTPS pages may wait for browser permission before reaching loopback.
		const health = await requestJson(`${LOCAL_INFERENCE_ORIGIN}/health`, signal, 60000);
		if (
			health.modelId !== GRANITE_MODEL ||
			health.revision !== GRANITE_REVISION ||
			health.maxTokens !== MAX_PROMPT_TOKENS
		) {
			throw new Error('Local Granite is running an incompatible model or service version.');
		}
		if (health.status === 'ready') break;
		if (health.status === 'error')
			throw new Error(health.error || 'Local Granite could not load the model.');
		if (health.status !== 'loading') throw new Error('Local Granite returned an unknown status.');
		if (Date.now() >= loadingDeadline)
			throw new Error(
				'Local Granite is still loading. Wait for the service to become ready, then retry.'
			);
		onProgress('loading');
		await pause(signal);
	}
	const job = await requestJson(`${LOCAL_INFERENCE_ORIGIN}/api/traces`, signal, 15000, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ prompt })
	});
	if (!validId(job.id) || !['queued', 'running'].includes(job.status))
		throw new Error('Local Granite returned an invalid inference job.');
	const statusUrl = localPath(job.statusUrl, `/api/traces/${job.id}`);
	onProgress(job.status);
	const deadline = Date.now() + JOB_TIMEOUT_MS;
	for (;;) {
		if (Date.now() >= deadline)
			throw new Error(
				'The inference run exceeded three minutes. Your previous diagram is still available; retry when the service is ready.'
			);
		const status = await requestJson(statusUrl, signal, 15000);
		if (status.id !== job.id) throw new Error('Local Granite returned a different inference job.');
		if (status.status === 'failed')
			throw new Error(status.error || 'The model could not generate this trace.');
		if (status.status === 'complete') {
			const assets = {
				main: localPath(status.mainUrl, `/traces/${job.id}/main.json`),
				attention: localPath(
					status.attentionUrl,
					`/traces/${job.id}/attention-operations-${job.id}.json`
				),
				expert: localPath(status.expertUrl, `/traces/${job.id}/expert-operations-${job.id}.json`)
			};
			onProgress('downloading');
			const document = validateLiveTrace(
				await requestJson(assets.main, signal, 60000),
				job.id,
				prompt
			);
			liveAssets.set(job.id, assets);
			trimLiveRecordings(liveAssets, 12);
			return document;
		}
		if (status.status !== 'queued' && status.status !== 'running')
			throw new Error('Local Granite returned an unknown inference state.');
		onProgress(status.status);
		await pause(signal);
	}
}
