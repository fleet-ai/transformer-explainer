import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const compiled = ts.transpileModule(
	readFileSync(new URL('../src/lib/moe/prompt.ts', import.meta.url), 'utf8'),
	{ compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }
).outputText;
const exports = {};
runInNewContext(compiled, { exports });
const { resolvePromptCompletion, preparePromptSubmission } = exports;

test('the bundled language prediction appends the actual recorded subword', () => {
	const recorded = JSON.parse(
		readFileSync(new URL('../static/traces/granite-moe.json', import.meta.url), 'utf8')
	);
	const example = recorded.examples.find((item) => item.id === 'language');
	const candidate = example.nextTokenCandidates[0];
	assert.equal(candidate.text, ' Par');
	const completion = resolvePromptCompletion(example.prompt, recorded.vocabulary, candidate.id);
	assert.deepEqual(
		{ ...completion },
		{
			text: 'The capital of France is Par',
			token: ' Par',
			error: '',
			finished: false
		}
	);
});

test('suffix and punctuation tokens join without invented spaces', () => {
	for (const [prompt, token, expected] of [
		['Par', 'is', 'Paris'],
		['Hello', ',', 'Hello,'],
		['answer', '.', 'answer.']
	]) {
		const completion = resolvePromptCompletion(prompt, [token], 0);
		assert.equal(completion.text, expected);
		assert.equal(completion.token, token);
	}
});

test('spaces, newlines and tabs remain exact tokenizer text', () => {
	for (const [prompt, token, expected] of [
		['Answer:', ' ', 'Answer: '],
		['Answer: ', '  ', 'Answer:   '],
		['def add(a, b):', '\n', 'def add(a, b):\n'],
		['def add(a, b):\n', '\t', 'def add(a, b):\n\t'],
		['  Café 🦉\n', '\t  answer', '  Café 🦉\n\t  answer']
	]) {
		const completion = resolvePromptCompletion(prompt, [token], 0);
		assert.equal(completion.text, expected);
		assert.equal(completion.token, token);
		assert.equal(completion.error, '');
	}
});

test('an edited draft replaces the old completion without whitespace changes', () => {
	const completion = resolvePromptCompletion('The capital of France is', [' Par'], 0);
	const draft = '  Café 🦉\n\tanswer: ';
	assert.equal(preparePromptSubmission(draft, completion, true), draft);
	assert.equal(
		preparePromptSubmission('The capital of France is', completion, false),
		completion.text
	);
});

test('successive predictions advance from the completed prompt without duplication', () => {
	const initial = 'The capital of France is';
	const first = resolvePromptCompletion(initial, [' Par', 'is'], 0);
	const firstPrompt = preparePromptSubmission(initial, first, false);
	assert.equal(firstPrompt, 'The capital of France is Par');
	// Retrying the same completion does not append it to an already-completed draft.
	assert.equal(preparePromptSubmission(firstPrompt, first, false), firstPrompt);
	const second = resolvePromptCompletion(firstPrompt, [' Par', 'is'], 1);
	assert.equal(
		preparePromptSubmission(firstPrompt, second, false),
		'The capital of France is Paris'
	);
});

test('missing vocabulary or token IDs leave the prompt unchanged', () => {
	for (const [vocabulary, tokenId] of [
		[undefined, 0],
		[['suffix'], undefined],
		[[], 0],
		[['suffix'], 4],
		[['suffix'], -1],
		[['suffix'], 0.5],
		[['suffix'], NaN]
	]) {
		assert.deepEqual(
			{ ...resolvePromptCompletion('prompt', vocabulary, tokenId) },
			{
				text: 'prompt',
				token: '',
				error: '',
				finished: false
			}
		);
	}
});

test('end-of-text finishes without inserting its control-token spelling', () => {
	assert.deepEqual(
		{ ...resolvePromptCompletion('Done.', ['<|endoftext|>'], 0) },
		{
			text: 'Done.',
			token: '',
			error: '',
			finished: true
		}
	);
});

test('partial Unicode token decoding is rejected instead of corrupting the prompt', () => {
	for (const token of ['\uFFFD', ' prefix\uFFFD', '\uFFFDsuffix']) {
		const completion = resolvePromptCompletion('Café', [token], 0);
		assert.equal(completion.text, 'Café');
		assert.equal(completion.token, '');
		assert.equal(completion.finished, false);
		assert.match(completion.error, /Cannot safely append this partial Unicode token/);
		assert.equal(preparePromptSubmission('Café 🦉', completion, true), 'Café 🦉');
	}
});
