import test from 'node:test';
import assert from 'node:assert/strict';
import { recordedAssetUrl } from '../src/lib/recorded-json.js';

const origin = 'https://raw.githubusercontent.com/fleet-ai/transformer-explainer/f3df6f1f3a275111e78491c9d5084a2867e3a1ad/static';
test('hosted full-model assets resolve to a pinned revision, including nested tensors and source parts', () => {
  for (const asset of ['deepseek-mla.json','attention-operations-deepseek-fact.json','attention-operations-deepseek-fact.json.part007.bin','expert-operations-deepseek-fact.json','t5-full.json','vit/block-00.f32','vlm/language/block-29.f32','hybrid/manifest.json']) {
    assert.equal(recordedAssetUrl(`/app/traces/${asset}`, origin), `${origin}/traces/${asset}`);
    assert.equal(recordedAssetUrl(`/traces/${asset}`, ''), `/traces/${asset}`);
  }
});
test('hosted asset routing preserves Granite, Dense and live inference locations', () => {
  for (const url of ['/traces/granite-moe.json','/traces/attention-operations-language.json','/traces/rope-example-128.json','/dense/','http://127.0.0.1:8766/traces/live-abc/main.json']) {
    assert.equal(recordedAssetUrl(url, origin), url);
  }
});
