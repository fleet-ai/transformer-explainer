import { spawnSync } from 'node:child_process';
import { readdir, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
// This immutable revision contains every full-model recording and its checksums.
const origin = 'https://raw.githubusercontent.com/fleet-ai/transformer-explainer/f3df6f1f3a275111e78491c9d5084a2867e3a1ad/static';
const build = spawnSync(process.execPath, [path.join(root, 'scripts/build-architectures.mjs')], {
  cwd: root, stdio: 'inherit', env: { ...process.env, VITE_RECORDED_TRACE_ORIGIN: origin }
});
if (build.error) throw build.error;
if (build.status !== 0) process.exit(build.status ?? 1);
const traces = path.join(root, 'build/traces');
for (const entry of await readdir(traces)) {
  if (/^(?:vit|vlm|hybrid|deepseek-mla\.json|(?:attention|expert)-operations-deepseek-fact\.json.*|t5-full\.json.*)$/.test(entry)) {
    await rm(path.join(traces, entry), { recursive: true, force: true });
  }
}
console.log(`Hosted recording origin: ${origin}`);
