import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';

const captured = JSON.parse(readFileSync(new URL('../static/traces/granite-moe.json', import.meta.url), 'utf8'));
const capturedAttention = JSON.parse(readFileSync(new URL('../static/traces/attention-operations-language.json', import.meta.url), 'utf8'));
const capturedExperts = JSON.parse(readFileSync(new URL('../static/traces/expert-operations-language.json', import.meta.url), 'utf8'));
const id = 'live-test123';
const prompt = captured.examples[0].prompt;
// Reuse real tensors to exercise the live one-example document contract.
const document = { ...captured, examples: [{ ...captured.examples[0], id }] };
const origin = 'http://127.0.0.1:8766';
const health = { status: 'ready', modelId: captured.model.id, revision: captured.model.revision, maxTokens: 32 };
const completed = {
  id, status: 'complete', mainUrl: `/traces/${id}/main.json`,
  attentionUrl: `/traces/${id}/attention-operations-${id}.json`,
  expertUrl: `/traces/${id}/expert-operations-${id}.json`
};
const compiled = new Map(['live', 'attention', 'operations'].map(name => [name, ts.transpileModule(
  readFileSync(new URL(`../src/lib/moe/${name}.ts`, import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }
).outputText]));
function production(fetch = () => { throw new Error('Unexpected fetch'); }) {
  const modules = new Map();
  function load(name) {
    if (modules.has(name)) return modules.get(name);
    const exports = {};
    modules.set(name, exports);
    runInNewContext(compiled.get(name), {
      exports, fetch, AbortController, DOMException, setTimeout, clearTimeout, Error, TypeError,
      require(dependency) {
        if (dependency === '$app/paths') return { base: '/moe' };
        if (dependency === './live') return load('live');
        throw new Error(`Unexpected dependency: ${dependency}`);
      }
    });
    return exports;
  }
  return { live: load('live'), attention: load('attention'), operations: load('operations') };
}
const response = data => ({ ok: true, status: 200, json: async () => data });

test('a genuine one-example capture is accepted and mismatched or missing tensors are rejected', () => {
  const { validateLiveTrace } = production().live;
  assert.equal(validateLiveTrace(document, id, prompt), document);
  for (const mutate of [
    data => { data.model.revision = 'other'; },
    data => { data.examples[0].prompt = 'different prompt'; },
    data => { data.examples[0].tokens[0].index = 3; },
    data => { data.examples[0].attentionScores[0][0][0][0] = NaN; },
    data => { data.examples[0].layers[0].queryPreview[0].pop(); },
    data => { data.examples[0].layers[0].tokens[0].selectedExperts[0] = 100; },
    data => { data.examples[0].nextTokenLogits.pop(); },
    data => { data.vocabulary.pop(); }
  ]) {
    const changed = structuredClone(document);
    mutate(changed);
    assert.throws(() => validateLiveTrace(changed, id, prompt), /trace/);
  }
});

test('prompt validation preserves exact code whitespace and blocks empty or excessive input', () => {
  const { promptError } = production().live;
  assert.match(promptError(' \n\t'), /Enter a prompt/);
  assert.match(promptError('a'.repeat(4097)), /4,096/);
  assert.equal(promptError('def add(a, b):\n    return a +'), '');
});

test('live tensor caches retain only three runs while preserving bundled recordings', () => {
  const { trimLiveRecordings } = production().live;
  const cache = new Map(['language', 'code', 'math', 'live-a', 'live-b', 'live-c', 'live-d'].map(id => [id, {}]));
  trimLiveRecordings(cache);
  assert.deepEqual([...cache.keys()], ['language', 'code', 'math', 'live-b', 'live-c', 'live-d']);
});

test('live inference loads matching main, attention and expert documents through the same registered job', async () => {
  const calls = [];
  const attention = { ...capturedAttention, example: { ...capturedAttention.example, id } };
  const expert = { ...capturedExperts, example: { ...capturedExperts.example, id } };
  const modules = production(async (url, init) => {
    calls.push([url, init]);
    if (url === `${origin}/health`) return response(health);
    if (url === `${origin}/api/traces`) {
      assert.equal(init.method, 'POST');
      assert.equal(JSON.parse(init.body).prompt, prompt);
      assert.equal(init.headers['X-MoE-Client'], 'explainer');
      return response({ id, status: 'queued', statusUrl: `/api/traces/${id}` });
    }
    if (url === `${origin}/api/traces/${id}`) return response(completed);
    if (url === `${origin}${completed.mainUrl}`) return response(document);
    if (url === `${origin}${completed.attentionUrl}`) return response(attention);
    if (url === `${origin}${completed.expertUrl}`) return response(expert);
    throw new Error(`Unexpected fetch ${url}`);
  });
  const progress = [];
  const result = await modules.live.generateLiveTrace(prompt, { signal: new AbortController().signal, onProgress: value => progress.push(value) });
  assert.equal(result, document);
  assert.deepEqual(progress, ['connecting', 'queued', 'downloading']);
  assert.equal(modules.live.traceAssetUrl(id, 'main', '/moe'), `${origin}${completed.mainUrl}`);
  assert.equal(await modules.attention.loadAttentionOperations(id), attention);
  assert.equal(await modules.operations.loadExpertOperations(id), expert);
  assert.equal(modules.live.traceAssetUrl('language', 'attention', '/moe'), '/moe/traces/attention-operations-language.json');
  assert.equal(modules.live.traceAssetUrl('language', 'main', '/moe'), '/moe/traces/granite-moe.json');
  assert.equal(calls.length, 6);
});

test('service responses cannot redirect prompt or trace requests to another host or job', async () => {
  for (const invalid of [
    { ...completed, mainUrl: 'https://example.com/trace.json' },
    { ...completed, attentionUrl: `/traces/other/attention-operations-${id}.json` },
    { ...completed, expertUrl: `//example.com/traces/${id}/expert-operations-${id}.json` }
  ]) {
    let calls = 0;
    const { live } = production(async () => response([health, { id, status: 'queued', statusUrl: `/api/traces/${id}` }, invalid][calls++]));
    await assert.rejects(live.generateLiveTrace(prompt, { signal: new AbortController().signal }), /unexpected trace URL/);
    assert.equal(calls, 3);
    assert.throws(() => live.traceAssetUrl(id, 'main'), /no longer connected/);
  }
});

test('server failures remain actionable and cancellation stops network work', async () => {
  const signal = new AbortController();
  signal.abort();
  await assert.rejects(production().live.generateLiveTrace(prompt, { signal: signal.signal }), { name: 'AbortError' });
  const offline = production(async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(offline.live.generateLiveTrace(prompt, { signal: new AbortController().signal }), /Start the local inference service/);
  let calls = 0;
  const limit = production(async () => calls++ === 0 ? response(health) : { ok: false, status: 422, json: async () => ({ error: 'Prompt has 40 tokens; use at most 32.' }) });
  await assert.rejects(limit.live.generateLiveTrace(prompt, { signal: new AbortController().signal }), /40 tokens; use at most 32/);
});
