import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { chunkTraceAssets } from '../scripts/chunk-trace-assets.mjs';
import { readRecordedJson } from '../src/lib/recorded-json.js';

test('static trace parts reconstruct identical JSON across UTF-8 boundaries', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'trace-parts-'));
  try {
    const document = { token: '世界 · Paris 🧠', coordinates: Array.from({ length: 200 }, (_, i) => i / 7) };
    await writeFile(path.join(directory, 'trace.json'), JSON.stringify(document));
    await chunkTraceAssets(directory, 100, 71);
    const manifest = JSON.parse(await readFile(path.join(directory, 'trace.json'), 'utf8'));
    assert.equal(manifest.kind, 'chunked-json');
    const response = { ok: true, url: 'https://example.test/base/traces/trace.json', json: async () => manifest };
    const fetchPart = async url => {
      assert.ok(url.startsWith('https://example.test/base/traces/'));
      return new Response(await readFile(path.join(directory, new URL(url).pathname.split('/').pop())));
    };
    assert.deepEqual(await readRecordedJson(response, fetchPart), document);
    const brokenPart = async url => {
      const bytes = new Uint8Array(await (await fetchPart(url)).arrayBuffer());
      bytes[0] ^= 1;
      return new Response(bytes);
    };
    await assert.rejects(readRecordedJson(response, brokenPart), /integrity check failed/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('ordinary recordings remain unchanged and unsafe chunk manifests fail', async () => {
  const original = { schemaVersion: 1, layers: [{ index: 0 }] };
  assert.deepEqual(await readRecordedJson(new Response(JSON.stringify(original))), original);
  await assert.rejects(readRecordedJson(new Response(JSON.stringify({ kind: 'chunked-json', chunks: [{ file: '../outside.bin' }] }))), /Invalid trace part/);
});
