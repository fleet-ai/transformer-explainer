<script lang="ts">
	import { Dropdown, DropdownItem, ButtonGroup } from 'flowbite-svelte';
	import Temperature from './Temperature.svelte';
	import Sampling from './Sampling.svelte';
	import { ChevronDownOutline } from 'flowbite-svelte-icons';
	import {
		inputText,
		predictedToken,
		rawPredictedToken,
		inputTextExample,
		isModelRunning,
		expandedBlock,
		selectedExampleIdx,
		weightPopover,
		denseStatus,
		denseError,
		denseSource,
		modelSession
	} from '~/store';
	import { generateDense, selectDenseExample } from '$lib/inference';
	import classNames from 'classnames';
	import LoadingDots from './common/LoadingDots.svelte';
	let inputRef: HTMLDivElement;
	let dropdownOpen = false;
	let draft = $inputText;
	let displayedPrompt = $inputText;
	$: if ($inputText !== displayedPrompt) {
		displayedPrompt = $inputText;
		draft = $inputText;
	}
	$: if (inputRef && inputRef.innerText !== draft) inputRef.innerText = draft;
	$: dirty = draft !== $inputText;
	$: disabled = $isModelRunning || $expandedBlock.id !== null || !!$weightPopover;
	$: invalid = !draft.trim()
		? 'Enter a prompt.'
		: draft.length > 4096
			? 'Use at most 4,096 characters and 32 tokens.'
			: '';
	function onInput() {
		draft = inputRef.innerText;
		$denseError = '';
	}
	function chooseExample(index: number) {
		dropdownOpen = false;
		selectDenseExample(index);
		draft = $inputText;
	}
	async function submit(event: Event) {
		event.preventDefault();
		if (disabled || invalid) return;
		if (!dirty && ($predictedToken?.tokenId === 50256 || $rawPredictedToken === '<|endoftext|>')) {
			$denseStatus = 'End of text · edit the prompt to continue';
			return;
		}
		if (!dirty && $rawPredictedToken.includes('\uFFFD')) {
			$denseError = 'Cannot safely append this partial Unicode token. Edit the prompt to continue.';
			return;
		}
		const prompt = dirty ? draft : draft + $rawPredictedToken;
		draft = prompt;
		try {
			await generateDense(prompt);
		} catch {
			/* The inference helper exposes an actionable error without replacing the previous diagram. */
		}
	}
	function keydown(event: KeyboardEvent) {
		event.stopPropagation();
		if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) void submit(event);
	}
	function paste(event: ClipboardEvent) {
		event.preventDefault();
		const selection = window.getSelection();
		if (!selection?.rangeCount) return;
		const range = selection.getRangeAt(0);
		range.deleteContents();
		const node = document.createTextNode(event.clipboardData?.getData('text/plain') ?? '');
		range.insertNode(node);
		range.setStartAfter(node);
		range.collapse(true);
		selection.removeAllRanges();
		selection.addRange(range);
		onInput();
	}
</script>

<div class="input-area" data-click="input-area">
	<form class="input-form" data-click="input-form" on:submit={submit}>
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
				{#each inputTextExample as text, index}<DropdownItem
						class={$selectedExampleIdx === index ? 'active' : undefined}
						on:click={() => chooseExample(index)}>{text}</DropdownItem
					>{/each}
			</Dropdown>
			<div
				class="input-container"
				data-click="text-input"
				class:disabled
				role="none"
				on:click={() => inputRef.focus()}
			>
				<div class="editable w-full">
					<div
						bind:this={inputRef}
						contenteditable={!disabled}
						class="text-box"
						role="textbox"
						aria-label="Prompt"
						aria-multiline="true"
						aria-disabled={disabled}
						aria-invalid={!!invalid}
						tabindex="0"
						title="Enter a prompt. Enter generates; Shift+Enter adds a line break. Up to 32 tokens."
						on:input={onInput}
						on:keydown={keydown}
						on:paste={paste}
					></div>
					{#if !$isModelRunning && !dirty}<div class="predicted" role="none">
							<span>{$predictedToken?.token ?? ''}</span>
						</div>{/if}
				</div>
				{#if $isModelRunning}<div class="loading"><LoadingDots /></div>{/if}
				<span class="helper-text" aria-live="polite"
					>{invalid ||
						($isModelRunning
							? $denseStatus
							: dirty
								? 'Draft · diagram shows the last completed prompt'
								: $denseStatus)}{#if !$isModelRunning && !invalid && !$modelSession}
						· first Generate downloads 626 MiB{/if}</span
				>
				{#if $denseError}<div class="input-error" role="alert">
						{$denseError}<button type="button" on:click|stopPropagation={() => ($denseError = '')}
							>Dismiss</button
						>
					</div>{/if}
			</div>
		</ButtonGroup>
		<button
			data-click="generate-btn"
			disabled={disabled || !!invalid}
			class={classNames('generate-button rounded-lg text-center text-sm shadow-sm', {
				disabled: disabled || !!invalid,
				active: !(disabled || invalid)
			})}
			type="submit">{$isModelRunning ? 'Running…' : 'Generate'}</button
		>
	</form>
	<div class="parameters" data-click="input-parameters">
		<Temperature {disabled} /><Sampling {disabled} />
	</div>
</div>

<style lang="scss">
	.input-error {
		position: absolute;
		top: calc(100% + 1.5rem);
		left: 0;
		z-index: 80;
		padding: 12px 15px;
		background: white;
		border: 1px solid #ddd5e5;
		border-radius: 6px;
		box-shadow: 0 4px 18px #0002;
		font-size: 12px;
		line-height: 1.5;
		white-space: normal;
		width: min(100%, 38rem);
	}
	.input-error button {
		display: block;
		margin-top: 6px;
		color: #8b5cf6;
	}

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
			overflow-y: hidden;
			justify-content: start;

			display: flex;
			align-items: center;
			overflow-x: hidden;
			white-space: nowrap;

			.text-box {
				min-width: 1ch;
				max-width: 100%;
				max-height: 1.35rem;
				line-height: 1.35rem;
				overflow: auto;
				scrollbar-width: none;
				white-space: pre;

				&:focus {
					outline: none;
				}
			}
			.predicted {
				flex: 1 0 0;
				color: var(--predicted-color);
				font-weight: 600;
				span {
					white-space: pre;
				}
			}
		}

		&.disabled {
			cursor: not-allowed;
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
