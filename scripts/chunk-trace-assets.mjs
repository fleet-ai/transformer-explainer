import { readFile, writeFile, readdir, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
// Keep each public asset below the static host's 25 MiB limit. Source traces
// remain unchanged; chunking only changes how the identical build bytes travel.
export async function chunkTraceAssets(directory, limit = 24 * 1024 * 1024, partSize = 8 * 1024 * 1024) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) { await chunkTraceAssets(file, limit, partSize); continue; }
    if (!entry.name.endsWith('.json') || (await stat(file)).size <= limit) continue;
    const bytes = await readFile(file), chunks = [];
    for (let start = 0; start < bytes.length; start += partSize) {
      const part = bytes.subarray(start, Math.min(bytes.length, start + partSize));
      const name = `${entry.name}.part${String(chunks.length).padStart(3, '0')}.bin`;
      await writeFile(path.join(directory, name), part);
      chunks.push({ file: name, bytes: part.length, sha256: hash(part) });
    }
    await writeFile(file, JSON.stringify({ kind: 'chunked-json', schemaVersion: 1, bytes: bytes.length, sha256: hash(bytes), chunks }) + '\n');
    console.log(`Split ${entry.name} into ${chunks.length} verified trace parts.`);
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await chunkTraceAssets(process.argv[2] || 'build/traces');
}
