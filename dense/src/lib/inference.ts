import { get } from 'svelte/store';
import { base } from '$app/paths';
import { ex0, ex1, ex2, ex3, ex4 } from '~/constants/examples';
import {
	modelData,
	tokens,
	tokenIds,
	inputText,
	predictedToken,
	rawPredictedToken,
	selectedExampleIdx,
	modelSession,
	isModelRunning,
	isFetchingModel,
	isLoaded,
	denseStatus,
	denseError,
	denseSource,
	temperature,
	sampling,
	blockIdx,
	blockIdxTemp,
	attentionHeadIdx,
	attentionHeadIdxTemp,
	expandedBlock,
	highlightedToken,
	hoveredMatrixCell
} from '~/store';
import { reshapeArray } from '~/utils/array';
import manifest from './model-manifest.json';

const examples = [ex0, ex1, ex2, ex3, ex4];
const TOKENIZER_REVISION = 'bf2c7f02e0b826c60d03af341171bde20893da66';
const MAX_TOKENS = 32;
let tokenizerPromise: Promise<any> | null = null;
let sessionPromise: Promise<any> | null = null;
let tokenizer: any;
let samplingRequest = 0;
const display = (text: string) =>
	text
		.replace(/\n/g, '[NEWLINE]')
		.replace(/\t/g, '[TAB]')
		.replace(/\r/g, '[CR]')
		.replace(/\s{2,}/g, (spaces) => `[${spaces.length} SPACES]`);

function resetDiagram() {
	blockIdx.set(0);
	blockIdxTemp.set(0);
	attentionHeadIdx.set(0);
	attentionHeadIdxTemp.set(0);
	expandedBlock.set({ id: null });
	highlightedToken.set({ index: null, value: null, fix: false });
	hoveredMatrixCell.set({ row: null, col: null });
}

export function selectDenseExample(index: number) {
	const example = examples[index];
	if (!example || get(isModelRunning)) return;
	samplingRequest++;
	const outputs = Object.fromEntries(
		Object.entries(example.outputs).map(([key, tensor]) => [
			key,
			{
				...tensor,
				data: tensor.data.map((row: (number | null)[]) =>
					row.map((value) => (value === null ? -Infinity : value))
				)
			}
		])
	);
	modelData.set({ ...example, outputs } as ModelData);
	tokens.set(example.tokens);
	tokenIds.set(example.tokenIds);
	inputText.set(example.prompt);
	predictedToken.set(example.sampled);
	rawPredictedToken.set(example.sampled.token);
	selectedExampleIdx.set(index);
	denseSource.set('recorded');
	denseStatus.set('Recorded GPT-2 example');
	denseError.set('');
	isLoaded.set(true);
	resetDiagram();
	if (
		tokenizer ||
		get(temperature) !== 0.8 ||
		get(sampling).type !== 'top-k' ||
		get(sampling).value !== 5
	)
		void resampleDense();
}

async function loadTokenizer() {
	if (!tokenizerPromise) {
		tokenizerPromise = (async () => {
			const { AutoTokenizer, env } = await import('@xenova/transformers');
			env.allowLocalModels = false;
			tokenizer = await AutoTokenizer.from_pretrained('Xenova/gpt2', {
				revision: TOKENIZER_REVISION
			});
			return tokenizer;
		})().catch((error) => {
			tokenizerPromise = null;
			throw new Error(
				`Could not load the GPT-2 tokenizer. Check your connection and retry. ${error instanceof Error ? error.message : ''}`
			);
		});
	}
	return tokenizerPromise;
}

/** Full-vocabulary filtering; display labels are never fed back into the tokenizer. */
function distribution(
	logits: ArrayLike<number>,
	codec: any
): { probabilities: Probabilities; sampled: Probability; raw: string } {
	const selectedTemperature = get(temperature),
		selectedSampling = get(sampling);
	const ranked = Array.from(logits, (logit, tokenId) => ({ logit, tokenId })).sort(
		(a, b) => b.logit - a.logit || a.tokenId - b.tokenId
	);
	const scaled = ranked.map((item) => item.logit / selectedTemperature);
	const exponentials = scaled.map((value) => Math.exp(value - scaled[0]));
	const mass = exponentials.reduce((sum, value) => sum + value, 0);
	let cumulative = 0;
	const all = exponentials.map((value) => {
		const probability = value / mass;
		cumulative += probability;
		return { probability, cumulative };
	});
	let cutoff =
		selectedSampling.type === 'top-k'
			? Math.min(ranked.length, Math.max(1, Math.floor(selectedSampling.value))) - 1
			: selectedSampling.value >= 1
				? ranked.length - 1
				: all.findIndex((item) => item.cumulative >= selectedSampling.value);
	if (cutoff < 0) cutoff = ranked.length - 1;
	const keptMass = exponentials.slice(0, cutoff + 1).reduce((sum, value) => sum + value, 0);
	const threshold = Math.random();
	let selectedRank = cutoff,
		selectedMass = 0;
	for (let index = 0; index <= cutoff; index++) {
		selectedMass += exponentials[index] / keptMass;
		if (threshold < selectedMass) {
			selectedRank = index;
			break;
		}
	}
	const row = (rank: number): Probability => ({
		...ranked[rank],
		rank,
		token: display(codec.decode([ranked[rank].tokenId], { clean_up_tokenization_spaces: false })),
		scaledLogit: scaled[rank],
		topKLogit: rank <= cutoff ? scaled[rank] : -Infinity,
		// The inspector shows exp(scaled logit); sampling above uses its stable shifted equivalent.
		expLogit: selectedSampling.type === 'top-k' && rank > cutoff ? 0 : Math.exp(scaled[rank]),
		probability: rank <= cutoff ? exponentials[rank] / keptMass : 0,
		topPProbability: all[rank].probability,
		cumulativeProbability: all[rank].cumulative,
		cutoffIndex: cutoff
	});
	return {
		probabilities: ranked.slice(0, 50).map((_, rank) => row(rank)),
		sampled: row(selectedRank),
		raw: codec.decode([ranked[selectedRank].tokenId], { clean_up_tokenization_spaces: false })
	};
}

export async function resampleDense() {
	const request = ++samplingRequest;
	const current = get(modelData);
	if (!current || get(isModelRunning)) return;
	try {
		const codec = await loadTokenizer();
		if (request !== samplingRequest || get(modelData) !== current) return;
		const result = distribution(current.logits, codec);
		modelData.set({ ...current, probabilities: result.probabilities, sampled: result.sampled });
		predictedToken.set(result.sampled);
		rawPredictedToken.set(result.raw);
	} catch (error) {
		if (request === samplingRequest)
			denseError.set(
				error instanceof Error ? error.message : 'Could not update sampling. Retry Generate.'
			);
	}
}

async function verifiedPart(
	part: (typeof manifest.parts)[number],
	cache: Cache | null,
	signal: AbortSignal
) {
	const url = manifest.baseUrl + part.name;
	const hex = (buffer: ArrayBuffer) =>
		Array.from(new Uint8Array(buffer), (value) => value.toString(16).padStart(2, '0')).join('');
	const valid = async (bytes: ArrayBuffer) =>
		bytes.byteLength === part.bytes &&
		hex(await crypto.subtle.digest('SHA-256', bytes)) === part.sha256;
	const cached = await cache?.match(url);
	if (cached) {
		const bytes = await cached.arrayBuffer();
		if (await valid(bytes)) return bytes;
		await cache?.delete(url);
	}
	const response = await fetch(url, { signal, credentials: 'omit' });
	if (!response.ok)
		throw new Error(
			`GPT-2 download failed (HTTP ${response.status}, ${part.name}). Retry Generate.`
		);
	const bytes = await response.arrayBuffer();
	if (!(await valid(bytes)))
		throw new Error(
			`GPT-2 model part ${part.name} does not match the original model. Retry later.`
		);
	try {
		await cache?.put(
			url,
			new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } })
		);
	} catch {
		/* Browser quota may prevent persistence; the verified model can still run. */
	}
	return bytes;
}

async function loadSession() {
	if (!sessionPromise) {
		sessionPromise = (async () => {
			isFetchingModel.set(true);
			denseStatus.set('Downloading GPT-2 · 0 / 63 parts · 626 MiB once');
			const controller = new AbortController();
			let timedOut = false;
			const timeout = setTimeout(() => {
				timedOut = true;
				controller.abort();
			}, 15 * 60_000);
			try {
				const cache =
					typeof caches !== 'undefined'
						? await caches.open(`dense-gpt2-${manifest.upstreamCommit}`).catch(() => null)
						: null;
				const buffers: ArrayBuffer[] = Array(manifest.parts.length);
				let next = 0,
					completed = 0;
				await Promise.all(
					Array.from({ length: 4 }, async () => {
						while (next < manifest.parts.length) {
							const index = next++;
							buffers[index] = await verifiedPart(manifest.parts[index], cache, controller.signal);
							completed++;
							denseStatus.set(`Downloading GPT-2 · ${completed} / 63 parts · 626 MiB once`);
						}
					})
				);
				const bytes = new Uint8Array(manifest.parts.reduce((sum, part) => sum + part.bytes, 0));
				let offset = 0;
				for (const buffer of buffers) {
					bytes.set(new Uint8Array(buffer), offset);
					offset += buffer.byteLength;
				}
				buffers.length = 0;
				denseStatus.set('Preparing GPT-2 for browser inference…');
				const ort = await import('onnxruntime-web');
				ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/';
				ort.env.wasm.numThreads = 1;
				ort.env.logLevel = 'error';
				const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
				modelSession.set(session);
				return session;
			} catch (error) {
				if (timedOut)
					throw new Error('GPT-2 download timed out. Retry Generate to reuse downloaded parts.');
				throw error;
			} finally {
				clearTimeout(timeout);
				controller.abort();
				isFetchingModel.set(false);
			}
		})().catch((error) => {
			sessionPromise = null;
			throw error;
		});
	}
	return sessionPromise;
}

/** Arbitrary input always runs the real browser model; cached examples are never substituted. */
export async function generateDense(prompt: string): Promise<void> {
	if (get(isModelRunning)) return;
	if (!prompt.trim()) throw new Error('Enter a prompt before generating.');
	if (prompt.length > 4096) throw new Error('Use at most 4,096 characters and 32 tokens.');
	denseError.set('');
	isModelRunning.set(true);
	samplingRequest++;
	try {
		denseStatus.set('Loading GPT-2 tokenizer…');
		const codec = await loadTokenizer();
		const ids: number[] = codec.encode(prompt);
		if (ids.length > MAX_TOKENS)
			throw new Error(`This prompt has ${ids.length} tokens. Use at most ${MAX_TOKENS}.`);
		const session = await loadSession();
		denseStatus.set('Running GPT-2 in this browser…');
		const ort = await import('onnxruntime-web');
		const results = await session.run({
			input: new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length])
		});
		const rawLogits = results.linear_output?.data;
		if (!rawLogits || rawLogits.length !== 50257)
			throw new Error('GPT-2 returned unexpected vocabulary logits.');
		const logits = Array.from(rawLogits as ArrayLike<number>);
		const outputs: ModelData['outputs'] = {};
		for (let layer = 0; layer < 12; layer++)
			for (let head = 0; head < 12; head++)
				for (const stage of [
					'attn',
					'attn_scaled',
					'attn_masked',
					'attn_softmax',
					'attn_dropout'
				]) {
					const name = `block_${layer}_attn_head_${head}_${stage}`;
					const tensor = results[name];
					if (!tensor) throw new Error(`GPT-2 did not return ${name}.`);
					outputs[name] = {
						data: reshapeArray(Array.from(tensor.data as ArrayLike<number>), tensor.dims),
						dims: tensor.dims,
						size: tensor.size
					};
				}
		const sampled = distribution(logits, codec);
		modelData.set({
			logits,
			outputs,
			probabilities: sampled.probabilities,
			sampled: sampled.sampled
		});
		tokens.set(
			ids.map((id) => display(codec.decode([id], { clean_up_tokenization_spaces: false })))
		);
		tokenIds.set(ids);
		inputText.set(prompt);
		predictedToken.set(sampled.sampled);
		rawPredictedToken.set(sampled.raw);
		selectedExampleIdx.set(-1);
		denseSource.set('inference');
		denseStatus.set('Fresh GPT-2 inference · in this browser');
		resetDiagram();
	} catch (error) {
		denseStatus.set('GPT-2 · retry available');
		const message = error instanceof Error ? error.message : 'Could not run GPT-2.';
		denseError.set(
			message.includes('fetch')
				? 'Could not download GPT-2. Check your connection, then retry Generate. Previously downloaded parts are reused.'
				: message
		);
		throw error;
	} finally {
		isModelRunning.set(false);
	}
}
