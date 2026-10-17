import { createInspector } from '$lib/inspectors';
import { writable, derived, readable } from 'svelte/store';
import type * as ort from 'onnxruntime-web';
import tailwindConfig from '../../tailwind.config';
import resolveConfig from 'tailwindcss/resolveConfig';
import { textPages } from '~/utils/textbookPages';
import type { TraceDocument, TraceExample } from '$lib/moe/types';
import type { AttentionOperationsDocument } from '$lib/moe/attention';

const { theme } = resolveConfig(tailwindConfig);

export const attentionHeadIdxTemp = writable(0);
export const attentionHeadIdx = writable(0);
export const blockIdx = writable(0);

export const isOnAnimation = writable(false);

// Textbook state management
export const textbookCurrentPage = writable<number>(0);
export const textbookPreviousPage = writable<number>(-1);
export const textbookCurrentPageId = writable<string>(textPages[0].id);
export const textbookPreviousPageId = writable<string>('');
export const isTextbookOpen = writable<boolean>(true);

// is transformer running?
export const isModelRunning = writable(false);
export const isFetchingModel = writable(true);
export const isLoaded = writable(false);

export const inputTextExample = [
	'The capital of France is',
	'def add(a, b):\n    return a +',
	'Question: What is 7 + 5?\nAnswer:'
];

const initialExIdx = 0;
export const selectedExampleIdx = writable<number>(initialExIdx);

export const modelSession = writable<ort.InferenceSession>();

// transformer model output
export const modelData = writable<ModelData>();
export const predictedToken = writable<Probability>();
export const tokens = writable<string[]>([]);
export const tokenIds = writable<number[]>([]);
export const moeDocument = writable<TraceDocument | null>(null);
export const moeExample = writable<TraceExample | null>(null);
export const moeModel = writable<TraceDocument['model'] | null>(null);
export const moeAttentionOperations = writable<AttentionOperationsDocument | null>(null);
export const moeTokenIndex = writable(0);

export const modelMetaMap: Record<string, ModelMetaData> = {
	'granite-moe': { layer_num: 24, attention_head_num: 16, dimension: 1024 },
	'deepseek-mla': { layer_num: 27, attention_head_num: 16, dimension: 2048 },
	gpt2: { layer_num: 12, attention_head_num: 12, dimension: 768, chunkTotal: 63 },
	'gpt2-medium': { layer_num: 24, attention_head_num: 16, dimension: 1024 },
	'gpt2-large': { layer_num: 36, attention_head_num: 20, dimension: 1280 }
};

// selected token vector
export const highlightedToken = writable<HighlightedToken>({
	index: null,
	value: null,
	fix: false
});

export const highlightedHead = writable<HighlightedToken>({
	index: null,
	value: null,
	fix: false
});

// expanded block
export const expandedBlock = writable<ExpandedBlock>({ id: null });
export const isExpandOrCollapseRunning = writable(false);

// user input text
export const inputText = writable(inputTextExample[initialExIdx]);
// export const tokens = derived(inputText, ($inputText) => $inputText.trim().split(' '));

// selected model and meta data
const initialSelectedModel = 'granite-moe';
export const selectedModel = writable(initialSelectedModel);
export const recordedControls = writable<{
	status?: string;
	traceUrl?: string;
	selectExample?: (index: number) => void;
	replay?: () => void;
} | null>(null);
export const modelMeta = derived([selectedModel, moeModel], ([$selectedModel, $moeModel]) =>
	!$selectedModel.startsWith('gpt2') && $moeModel
		? {
				layer_num: $moeModel.numLayers,
				attention_head_num: $moeModel.attentionHeads,
				dimension: $moeModel.hiddenSize
			}
		: modelMetaMap[$selectedModel]
);

// Temperature setting
export const initialTemperature = 1;
export const temperature = writable(initialTemperature);

// Sampling
export const sampling = writable<Sampling>({ type: 'top-p', value: 1 });

// Prediction visual
export const highlightedIndex = writable(null);
export const finalTokenIndex = writable(null);

// Visual element style
export const rootRem = 16;
export const minVectorHeight = 12;
export const maxVectorHeight = 30;
export const maxVectorScale = 3.4;

export const vectorHeight = writable(0);
export const headContentHeight = writable(0);
export const headGap = { x: 5, y: 8, scale: 0 };

export const isBoundingBoxActive = writable(false);

export const predictedColor = theme.colors.purple[600];

// Interactivity
export const hoveredPath = writable();
export const hoveredMatrixCell = writable({ row: null, col: null });
export const weightPopover = createInspector<any>();
export const tooltip = writable();

export const isMobile = readable(false, (set) => {
	if (typeof window !== 'undefined') {
		// Only run in browser environment
		const userAgent = navigator.userAgent.toLowerCase();
		set(/android|iphone|ipad|ipod/i.test(userAgent));
	}
	return () => {}; // Cleanup function
});

// User identification
export const userId = writable<string | null>(null);
