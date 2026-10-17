import { createServer } from 'node:http';
import { open, realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const buildDirectory = fileURLToPath(new URL('../build/', import.meta.url));
const mimeTypes = {
	'.html': 'text/html; charset=utf-8',
	'.js': 'text/javascript; charset=utf-8',
	'.mjs': 'text/javascript; charset=utf-8',
	'.css': 'text/css; charset=utf-8',
	'.json': 'application/json; charset=utf-8',
	'.map': 'application/json; charset=utf-8',
	'.svg': 'image/svg+xml',
	'.png': 'image/png',
	'.jpg': 'image/jpeg',
	'.jpeg': 'image/jpeg',
	'.gif': 'image/gif',
	'.webp': 'image/webp',
	'.avif': 'image/avif',
	'.ico': 'image/x-icon',
	'.wasm': 'application/wasm',
	'.woff': 'font/woff',
	'.woff2': 'font/woff2',
	'.ttf': 'font/ttf',
	'.otf': 'font/otf',
	'.txt': 'text/plain; charset=utf-8',
	'.xml': 'application/xml; charset=utf-8'
};

function within(root, candidate) {
	const relative = path.relative(root, candidate);
	return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

/** Serve the combined static output; no SPA fallback or directory listings. */
export async function createPreviewServer(directory = buildDirectory) {
	const root = await realpath(directory);
	if (!(await stat(root)).isDirectory()) throw new Error('Preview requires a build directory.');
	return createServer(async (request, response) => {
		response.setHeader('X-Content-Type-Options', 'nosniff');
		response.setHeader('Cache-Control', 'no-store');
		const reply = (status, text) => {
			response.writeHead(status, {
				'Content-Type': 'text/plain; charset=utf-8',
				'Content-Length': Buffer.byteLength(text)
			});
			response.end(request.method === 'HEAD' ? undefined : text);
		};
		if (request.method !== 'GET' && request.method !== 'HEAD') {
			response.setHeader('Allow', 'GET, HEAD');
			reply(405, 'Method not allowed.');
			return;
		}
		let handle;
		try {
			// Inspect before URL normalization so encoded traversal cannot disappear.
			const rawPath = (request.url ?? '').split('?')[0];
			const decoded = decodeURIComponent(rawPath);
			if (
				!decoded.startsWith('/') ||
				decoded.startsWith('//') ||
				decoded.includes('\\') ||
				decoded.includes('\0') ||
				decoded.split('/').some((part) => part.startsWith('.'))
			) {
				reply(404, 'Not found.');
				return;
			}
			let filename = await realpath(path.join(root, decoded.slice(1)));
			if (!within(root, filename)) {
				reply(404, 'Not found.');
				return;
			}
			if ((await stat(filename)).isDirectory()) {
				filename = await realpath(path.join(filename, 'index.html'));
				if (!within(root, filename)) {
					reply(404, 'Not found.');
					return;
				}
				if (!rawPath.endsWith('/')) {
					const query = request.url.slice(rawPath.length);
					response.setHeader('Location', `${rawPath}/${query}`);
					reply(308, 'Use the directory URL.');
					return;
				}
			}
			handle = await open(filename, 'r');
			const info = await handle.stat();
			if (!info.isFile()) {
				await handle.close();
				handle = undefined;
				reply(404, 'Not found.');
				return;
			}
			response.writeHead(200, {
				'Content-Type':
					mimeTypes[path.extname(filename).toLowerCase()] ?? 'application/octet-stream',
				'Content-Length': info.size
			});
			if (request.method === 'HEAD') {
				await handle.close();
				handle = undefined;
				response.end();
			} else {
				await pipeline(handle.createReadStream(), response);
				handle = undefined;
			}
		} catch (error) {
			await handle?.close().catch(() => {});
			if (response.headersSent) {
				response.destroy();
			} else {
				const missing =
					error instanceof URIError || ['ENOENT', 'ENOTDIR', 'ELOOP'].includes(error.code);
				reply(missing ? 404 : 500, missing ? 'Not found.' : 'Could not read the built file.');
			}
		}
	});
}

async function main() {
	const { values } = parseArgs({
		options: {
			host: { type: 'string', default: '127.0.0.1' },
			port: { type: 'string', default: '4173' }
		}
	});
	if (!/^\d+$/.test(values.port) || Number(values.port) > 65535) {
		throw new Error('--port must be an integer from 0 to 65535.');
	}
	const server = await createPreviewServer();
	await new Promise((resolve, reject) => {
		server.once('error', reject);
		server.listen(Number(values.port), values.host, resolve);
	});
	const host = values.host.includes(':') ? `[${values.host}]` : values.host;
	const origin = `http://${host}:${server.address().port}`;
	console.log(`MoE: ${origin}/\nDense: ${origin}/dense/`);
	for (const signal of ['SIGINT', 'SIGTERM']) {
		process.once(signal, () => server.close());
	}
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
	main().catch((error) => {
		console.error(`Preview failed: ${error.message}\nRun npm run build before starting preview.`);
		process.exitCode = 1;
	});
}
