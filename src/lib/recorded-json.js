const digest = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), value => value.toString(16).padStart(2, '0')).join('');

/** Load either ordinary trace JSON or lossless parts emitted by the static build. */
export async function readRecordedJson(response, fetchPart = fetch) {
  if (!response.ok) throw new Error(`Recording returned HTTP ${response.status}.`);
  const document = await response.json();
  if (document.kind !== 'chunked-json') return document;
  if (document.schemaVersion !== 1 || !Number.isSafeInteger(document.bytes) || document.bytes < 1 || document.bytes > 512 * 1024 * 1024 ||
      !Array.isArray(document.chunks) || document.chunks.length < 1 || document.chunks.length > 128 ||
      document.chunks.some(part => !/^[\w.-]+\.bin$/.test(part.file) || !Number.isSafeInteger(part.bytes) || part.bytes < 1 || part.bytes > 24 * 1024 * 1024 || !/^[a-f0-9]{64}$/.test(part.sha256)) ||
      document.chunks.reduce((sum, part) => sum + part.bytes, 0) !== document.bytes) {
    throw new Error('Invalid trace part manifest.');
  }
  const result = new Uint8Array(document.bytes);
  let offset = 0;
  // Four requests at a time bound memory and network contention for large traces.
  for (let start = 0; start < document.chunks.length; start += 4) {
    const parts = await Promise.all(document.chunks.slice(start, start + 4).map(async part => {
      const reply = await fetchPart(new URL(part.file, response.url).href);
      if (!reply.ok) throw new Error(`Trace part returned HTTP ${reply.status}.`);
      const bytes = new Uint8Array(await reply.arrayBuffer());
      if (bytes.length !== part.bytes || await digest(bytes) !== part.sha256) throw new Error('Trace part integrity check failed.');
      return bytes;
    }));
    for (const part of parts) { result.set(part, offset); offset += part.length; }
  }
  if (await digest(result) !== document.sha256) throw new Error('Complete trace integrity check failed.');
  return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(result));
}

/** Hosted builds read large immutable recordings from the Fleet source revision. */
export function recordedAssetUrl(url, origin = import.meta.env?.VITE_RECORDED_TRACE_ORIGIN || '') {
  if (!origin || !url.startsWith('/')) return url;
  const start = url.indexOf('/traces/');
  if (start < 0) return url;
  const asset = url.slice(start + '/traces/'.length);
  if (!/^(?:deepseek-mla\.json$|(?:attention|expert)-operations-deepseek-fact\.json(?:\.part\d+\.bin)?$|t5-full\.json$|(?:vit|vlm|hybrid)\/)/.test(asset)) return url;
  return origin.replace(/\/$/, '') + url.slice(start);
}
