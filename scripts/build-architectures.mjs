import { spawnSync } from 'node:child_process';
import { cp, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { chunkTraceAssets } from './chunk-trace-assets.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const vite = path.join(root, 'node_modules/vite/bin/vite.js');

// Each architecture owns its model stores and runtime. Both ship at the same
// origin, with the original dense application mounted beneath /dense/.
for (const directory of [root, path.join(root, 'dense')]) {
	const result = spawnSync(process.execPath, [vite, 'build'], {
		cwd: directory,
		stdio: 'inherit',
		env: process.env
	});
	if (result.error) throw result.error;
	if (result.status !== 0) process.exit(result.status ?? 1);
}

const destination = path.join(root, 'build/dense');
await rm(destination, { recursive: true, force: true });
await cp(path.join(root, 'dense/build'), destination, { recursive: true });
await chunkTraceAssets(path.join(root, 'build/traces'));
console.log('Built Dense at /dense/ and MoE at /.');
