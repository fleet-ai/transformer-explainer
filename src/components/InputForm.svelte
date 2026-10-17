<script lang="ts">
	import { onDestroy, tick } from 'svelte';
	import { Dropdown, DropdownItem, ButtonGroup } from 'flowbite-svelte';
	import Temperature from './Temperature.svelte';
	import Sampling from './Sampling.svelte';
	import { ChevronDownOutline } from 'flowbite-svelte-icons';
	import {
		inputText,
		predictedToken,
		isModelRunning,
		expandedBlock,
		weightPopover,
		moeDocument,
		moeExample,
		moeModel,
		selectedExampleIdx,
		recordedControls
	} from '~/store';
	import { hydrateRecordedExample } from '$lib/moe/adapter';
	import {
		generateLiveTrace,
		isLiveTrace,
		LOCAL_INFERENCE_ORIGIN,
		promptError,
		traceAssetUrl,
		type LiveProgress
	} from '$lib/moe/live';
	import type { TraceDocument } from '$lib/moe/types';
	import { resolvePromptCompletion, preparePromptSubmission } from '$lib/moe/prompt';
	import { base } from '$app/paths';
	import classNames from 'classnames';
	import LoadingDots from './common/LoadingDots.svelte';
	let inputRef: HTMLTextAreaElement;
	let previewRef: HTMLDivElement;
	let editing = false;
	let dropdownOpen = false;
	let draft = $inputText || '';
	let displayedPrompt = $inputText;
	let presetDocument: TraceDocument | null = null;
	let controller: AbortController | null = null;
	let busy = false;
	let error = '';
	let serviceState = 'connect on Generate';
	let showSetup = false;
	let progress: LiveProgress = 'connecting';
	const progressLabels: Record<LiveProgress, string> = {
		connecting: 'Connecting to Local Granite… Allow local-network access if prompted.',
		loading: 'Loading Granite on this Mac…',
		queued: 'Waiting for the local model…',
		running: 'Running Granite and recording every block…',
		downloading: 'Loading the fresh trace…'
	};
	$: if ($inputText !== displayedPrompt) {
		if (!editing) draft = $inputText;
		displayedPrompt = $inputText;
	}
	$: if ($moeDocument && !$moeDocument.examples.some((example) => isLiveTrace(example.id)))
		presetDocument = $moeDocument;
	$: isMLA = $moeModel?.architecture === 'mla';
	$: recordedOnly = !!$moeModel?.architecture && $moeModel.architecture !== 'moe';
	$: completion = resolvePromptCompletion(
		$inputText,
		$moeDocument?.vocabulary,
		$predictedToken?.tokenId
	);
	$: dirty = editing || draft !== $inputText;
	$: sourcePromptOnly =
		$moeModel?.architecture === 'encoder-decoder' || $moeModel?.outputKind === 'classification';
	$: editorText = sourcePromptOnly ? draft : preparePromptSubmission(draft, completion, dirty);
	$: showPrediction = !sourcePromptOnly && !busy && !dirty && !!completion.token;
	$: if (inputRef && editorText && !dirty) revealCompletion(editorText);
	$: disabled = !$moeDocument || busy;
	$: parameterDisabled =
		!$moeDocument || !!$weightPopover || busy || $moeModel?.outputKind === 'classification';
	$: invalidPrompt = recordedOnly
		? ''
		: promptError(editorText) || (!dirty ? completion.error : '');
	$: finished = !dirty && completion.finished;
	$: traceUrl =
		$recordedControls?.traceUrl ??
		(isMLA
			? `${base}/traces/deepseek-mla.json`
			: traceAssetUrl($moeExample?.id ?? '', 'main', base));
	$: currentIsLive = isLiveTrace($moeExample?.id ?? '');
	function closeDiagramDetails() {
		$expandedBlock = { id: null };
		$weightPopover = null;
	}
	function onSelectExample(index: number) {
		if (busy || !presetDocument) return;
		dropdownOpen = false;
		error = '';
		editing = false;
		draft = presetDocument.examples[index].prompt;
		closeDiagramDetails();
		if ($recordedControls?.selectExample) $recordedControls.selectExample(index);
		else hydrateRecordedExample(presetDocument, index);
	}
	async function handleSubmit(event: Event) {
		event.preventDefault();
		if (disabled || (!recordedOnly && finished)) return;
		if (recordedOnly) {
			const index = dirty
				? (presetDocument?.examples.findIndex(
						(example) => example.prompt.trim() === draft.trim()
					) ?? -1)
				: $selectedExampleIdx;
			if (index < 0) {
				error = `This ${$moeModel?.name ?? 'model'} view contains complete recorded forward passes. Choose a saved example to replay every layer.`;
				return;
			}
			if (!dirty && $recordedControls?.replay) $recordedControls.replay();
			else onSelectExample(index);
			return;
		}
		if (invalidPrompt) {
			error = invalidPrompt;
			return;
		}
		closeDiagramDetails();
		const submittedPrompt = editorText;
		// Accept the visible token once. Retries then reuse this exact draft.
		draft = submittedPrompt;
		editing = true;
		controller?.abort();
		const request = new AbortController();
		controller = request;
		busy = true;
		$isModelRunning = true;
		error = '';
		try {
			const document = await generateLiveTrace(submittedPrompt, {
				signal: request.signal,
				onProgress: (state) => {
					progress = state;
					serviceState =
						state === 'connecting' ? 'connecting' : state === 'loading' ? 'loading' : 'connected';
				}
			});
			if (request.signal.aborted) return;
			hydrateRecordedExample(document);
			draft = submittedPrompt;
			editing = false;
			serviceState = 'ready';
		} catch (reason) {
			if (!request.signal.aborted) {
				error =
					reason instanceof Error ? reason.message : 'Could not generate a trace. Please retry.';
				serviceState = progress === 'connecting' ? 'unavailable' : 'retry available';
			}
		} finally {
			if (controller === request) {
				controller = null;
				busy = false;
				$isModelRunning = false;
			}
		}
	}
	function handleKeyDown(event: KeyboardEvent) {
		event.stopPropagation();
		if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) handleSubmit(event);
	}
	function onInput() {
		draft = inputRef.value;
		editing = true;
		error = '';
	}
	function onFocusInput() {
		closeDiagramDetails();
		if (!dirty) draft = recordedOnly ? $inputText : completion.text;
		editing = true;
	}
	function syncPreviewScroll() {
		const preview = previewRef;
		if (!preview || !inputRef) return;
		const atRightEdge = inputRef.scrollLeft >= inputRef.scrollWidth - inputRef.clientWidth - 1;
		const boldOverflow = atRightEdge ? Math.max(0, preview.scrollWidth - inputRef.scrollWidth) : 0;
		preview.scrollLeft = inputRef.scrollLeft + boldOverflow;
		preview.scrollTop = inputRef.scrollTop;
	}
	async function revealCompletion(text: string) {
		await tick();
		const editor = inputRef;
		if (!editor || dirty || editorText !== text) return;
		editor.scrollTop = editor.scrollHeight;
		// Follow the final line, even when an earlier code line is much longer.
		const preview = previewRef;
		const fragments = preview?.lastElementChild?.getClientRects();
		const tail = fragments?.item(fragments.length - 1);
		const end =
			preview && tail ? tail.right - preview.getBoundingClientRect().left + preview.scrollLeft : 0;
		editor.scrollLeft = Math.max(0, end - editor.clientWidth + 2);
		syncPreviewScroll();
	}
	onDestroy(() => {
		controller?.abort();
		$isModelRunning = false;
	});
</script>

<div class="input-area" data-click="input-area">
	<form class="input-form" data-click="input-form" on:submit={handleSubmit}>
		<ButtonGroup class="input-btn-group" size="sm">
			<button
				data-click="dropdown-btn"
				type="button"
				{disabled}
				class:selectDisabled={disabled}
				class="select-button inline-flex shrink-0 items-center justify-center border border-s-0 border-gray-200 bg-white px-3 py-2 text-center text-xs font-medium text-gray-900 first:rounded-s-lg first:border-s last:rounded-e-lg"
			>
				Examples<ChevronDownOutline class="pointer-events-none h-4 w-4 text-gray-500" />
			</button>
			<Dropdown bind:open={dropdownOpen} class="example-dropdown">
				{#each presetDocument?.examples ?? [] as example, index}
					<DropdownItem
						data-click={`dropdown-item-${index}`}
						class={$moeExample?.id === example.id ? 'active' : undefined}
						on:click={() => onSelectExample(index)}>{example.prompt}</DropdownItem
					>
				{/each}
			</Dropdown>
			<div
				data-click="text-input"
				class="input-container"
				class:disabled={busy}
				role="none"
				on:click={() => inputRef.focus()}
			>
				<div class="editable">
					<textarea
						bind:this={inputRef}
						value={editorText}
						disabled={busy}
						rows="1"
						class="text-box"
						class:has-prediction={showPrediction}
						placeholder="Type your own prompt…"
						title={recordedOnly
							? 'Choose a recorded prompt from Examples. Replay returns to its first transformer block.'
							: 'Enter a prompt. Enter generates; Shift+Enter inserts a line break. Up to 32 tokens.'}
						aria-label="Prompt"
						aria-invalid={!!invalidPrompt}
						aria-describedby="prompt-status"
						on:input={onInput}
						on:focus={onFocusInput}
						on:scroll={syncPreviewScroll}
						on:keydown={handleKeyDown}
					></textarea>
					{#if showPrediction}<div class="prompt-preview" bind:this={previewRef} aria-hidden="true">
							<span>{$inputText}</span><span class="predicted">{completion.token}</span>
						</div>{/if}
				</div>
				{#if busy}<div class="loading"><LoadingDots /></div>{/if}
				<div class="helper-text" id="prompt-status" aria-live="polite">
					{#if busy}{progressLabels[progress]}
					{:else if recordedOnly}{$recordedControls?.status ??
							`${$moeModel?.name} · full ${$moeModel?.numLayers}-block inference`} ·
						<a href={traceUrl} download>trace JSON ↓</a>
					{:else if invalidPrompt}<span class="draft-status">{invalidPrompt}</span>
					{:else if finished}End of text · edit the prompt to continue
					{:else if dirty}<span class="draft-status"
							>Draft · diagram shows the last completed prompt</span
						>
					{:else}Click the prompt to edit · {currentIsLive
							? 'Fresh Granite trace'
							: 'Recorded Granite MoE'} ·
						<a href={traceUrl} download>trace JSON ↓</a>{/if}
					{#if !recordedOnly}<span class="local-status">
							· Local Granite: {serviceState} ·
							<button
								type="button"
								on:click|stopPropagation={() => (showSetup = !showSetup)}
								aria-expanded={showSetup}>setup</button
							></span
						>{/if}
				</div>
				{#if error || showSetup}
					<div class="prompt-feedback" on:click|stopPropagation role="none">
						{#if error}<p role="alert">{error}</p>{/if}
						{#if serviceState === 'unavailable' || showSetup}
							<p>
								<a href={LOCAL_INFERENCE_ORIGIN} target="_blank" rel="noopener">Open local app ↗</a
								>
								to use this same diagram directly with the model on your Mac.
							</p>
						{/if}
						{#if showSetup}
							<p>
								Prompts run on this Mac with the same Granite model. Up to 32 tokens; Enter
								generates and Shift+Enter adds a line break.
							</p>
							<p>Start the local inference service from this project’s Python environment:</p>
							<code>python scripts/serve_traces.py --cache-dir /private/tmp/moe-hf-cache</code>
							<p>
								Allow local-network access if your browser asks. If it cannot connect, open the
								local app above. Recorded examples remain available when the service is offline.
							</p>
						{/if}
						<button
							type="button"
							on:click={() => {
								error = '';
								showSetup = false;
							}}>Dismiss</button
						>
					</div>
				{/if}
			</div>
		</ButtonGroup>
		<button
			data-click="generate-btn"
			disabled={disabled || !!invalidPrompt || (!recordedOnly && finished)}
			class={classNames('generate-button rounded-lg text-center text-sm shadow-sm', {
				disabled: disabled || !!invalidPrompt || (!recordedOnly && finished),
				active: !(disabled || invalidPrompt || (!recordedOnly && finished))
			})}
			title={recordedOnly
				? 'Replay the full recorded model forward pass'
				: invalidPrompt ||
					(finished
						? 'End of text. Edit the prompt to continue.'
						: 'Continue this text on Local Granite and record its trace')}
			type="submit">{busy ? 'Running…' : recordedOnly ? 'Replay' : 'Generate'}</button
		>
	</form>
	{#if $moeModel?.outputKind !== 'classification'}<div
			class="parameters"
			data-click="input-parameters"
		>
			<Temperature disabled={parameterDisabled} /><Sampling disabled={parameterDisabled} />
		</div>{/if}
</div>

<style lang="scss">
	.parameters {
		display: flex;
		gap: 1rem;
	}
	.input-area {
		width: 100%;
		flex-shrink: 0;
		display: flex;
		gap: 1rem;
		padding-left: 1rem;
		padding-right: 1.25rem;

		.input-form {
			width: 100%;
			flex: 1 0 0;
			display: flex;
			align-items: center;
			gap: 0.5rem;

			:global(.input-btn-group) {
				flex: 1 0 0;
				display: flex;
			}
		}
	}
	.input-container {
		position: relative;
		display: flex;
		flex: 1 0 0;
		align-items: center;

		border: 1px solid theme('colors.gray.300');
		color: theme('colors.gray.900');
		border-left: none;
		border-start-end-radius: 0.5rem;
		border-end-end-radius: 0.5rem;
		font-size: 1rem;
		line-height: 1rem;
		padding: 0.5rem;
		white-space: pre-wrap;
		gap: 0.3rem;

		width: 10px; // to keep input box width

		.editable {
			position: relative;
			flex: 1 1 auto;
			min-width: 0;
			width: 100%;
			height: 1.35rem;
			line-height: 1.35rem;

			.text-box,
			.prompt-preview {
				width: 100%;
				height: 100%;
				border: 0;
				border-radius: 0;
				padding: 0;
				margin: 0;
				background: transparent;
				color: inherit;
				font: inherit;
				line-height: inherit;
				white-space: pre;
				text-align: left;
				tab-size: 8;
			}
			.text-box {
				display: block;
				overflow: auto;
				scrollbar-width: none;
				resize: none;
				&:focus {
					outline: none;
					box-shadow: none;
				}
				&.has-prediction {
					color: transparent;
					caret-color: theme('colors.gray.900');
				}
			}
			.prompt-preview {
				position: absolute;
				inset: 0;
				overflow: hidden;
				pointer-events: none;
			}
			.predicted {
				color: var(--predicted-color);
				font-weight: 600;
			}
		}

		&.disabled {
			cursor: not-allowed;
		}
		&:focus-within {
			border-color: var(--predicted-color);
			box-shadow: 0 0 0 1px var(--predicted-color);
		}

		.loading {
			flex-shrink: 0;
		}
	}

	.select-button {
		flex-shrink: 0;
		font-size: 0.9rem;
		border: 1px solid theme('colors.gray.300');
		color: theme('colors.gray.900');
		&:hover {
			background-color: theme('colors.gray.100');
		}
		&:focus {
			outline: none;
		}
		&.disabled {
			cursor: not-allowed;
		}
	}
	:global(.example-dropdown) {
		:global(.active) {
			background-color: theme('colors.gray.100') !important;
		}
	}
	.prompt-feedback {
		position: absolute;
		z-index: 60;
		top: calc(100% + 1.8rem);
		left: 0;
		width: min(38rem, 100%);
		padding: 0.85rem 1rem;
		border: 1px solid #ddd6e8;
		border-radius: 0.6rem;
		background: white;
		box-shadow: 0 4px 14px #0002;
		color: #4b5563;
		white-space: normal;
		line-height: 1.5;
		font-size: 0.85rem;
		p + p,
		code + p {
			margin-top: 0.5rem;
		}
		code {
			display: block;
			margin-top: 0.4rem;
			overflow-wrap: anywhere;
			font-size: 0.75rem;
			color: #6d28d9;
		}
		button {
			margin-top: 0.5rem;
			color: #7c3aed;
		}
	}
	.local-status {
		color: #9ca3af;
	}
	.helper-text button {
		text-decoration: underline;
	}
	.draft-status {
		color: #7c3aed;
	}
	.helper-text {
		position: absolute;
		bottom: 0;
		right: 0;
		padding: 0.3rem 0;
		transform: translate(0, 100%);
		color: theme('colors.gray.400');
		font-size: 0.72rem;
		white-space: nowrap;
	}
	:global(.generate-button) {
		padding: 0.4rem 0.8rem;
		border: 1px solid theme('colors.gray.300');
		color: theme('colors.gray.900');
		transition: all 0.2s;
		flex-shrink: 0;
	}
	:global(.generate-button.active) {
		&:hover {
			border: 1px solid var(--predicted-color);
			color: var(--predicted-color);
		}
	}
	:global(.generate-button.disabled) {
		opacity: 1;
		background-color: theme('colors.gray.100');
		color: theme('colors.gray.400');
		cursor: not-allowed;
	}

	:global(.generate-button):focus {
		border: 1px solid var(--predicted-color);
		color: var(--predicted-color);
	}
</style>
