# Complete recorded architecture views

The five additional tabs execute complete pretrained models and replay their
recorded activations in the original Transformer Explainer diagram components.
The site ships traces, including limited weight rows used for arithmetic, rather
than complete checkpoints. Dense and Granite retain their existing generation
runtimes; the five additional tabs replay bundled examples.

| Tab | Checkpoint | Complete computation |
| --- | --- | --- |
| MLA | [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | 27 blocks, all 16 query heads, dense first FFN, 26 routed/shared MoE layers, final RMSNorm and 102,400 LM logits |
| Encoder–decoder | [T5-small](https://huggingface.co/google-t5/t5-small) | 6 encoder + 6 decoder blocks, every one of 8 heads in self/cross attention, full translation generation and next-token logits |
| Mamba hybrid | [Zamba2-1.2B](https://huggingface.co/Zyphra/Zamba2-1.2B) | 38 Mamba2 blocks, six of which also call the shared Transformer with distinct adapters; 32 attention heads, 64 SSM heads, final RMSNorm and 32,000 LM logits |
| Vision–language | [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | 12 vision blocks, all 12 heads, pixel shuffle/projector, 30 language blocks with 9 Q / 3 KV heads, complete vocabulary output and generated completion |
| ViT | [DeiT-tiny-patch16-224](https://huggingface.co/facebook/deit-tiny-patch16-224) | 12 blocks, all 3 heads, 196 patch tokens + class token, final normalization and ImageNet classifier |

Each manifest records the checkpoint revision, hashes, capture script and library
versions. Model predictions are preserved, including incorrect labels or generated
descriptions. These examples illustrate architecture, not comparative model quality.

## Trace formats and numeric scope

- MLA uses `static/traces/deepseek-mla.json` with attention/expert operation
  sidecars. The attention sidecar contains all content Q/K, rotary Q/K before and
  after rotation, the compressed KV latent, all V heads, contexts and attention
  output. Expert inspectors show four real weight rows/columns plus the recorded
  contribution of omitted inputs, as in Granite.
- T5 uses `static/traces/t5-full.json`, with both complete stacks and rectangular
  cross-attention. T5 uses relative attention bias and no RoPE. The decoder output
  follows its actual generated translation prefix.
- ViT, VLM and hybrid use `static/traces/{vit,vlm,hybrid}/manifest.json` and
  little-endian float32 binary files. Each tensor has an element offset, length
  and shape; each file has its byte length and SHA-256. Binary storage preserves
  captured float32 values. Large views load the selected block on demand.

Production builds split JSON assets larger than 24 MiB into verified 8 MiB parts
to fit the static host's per-file limit. The original URL then returns a
`chunked-json` manifest; the browser verifies and reassembles the identical bytes
before parsing them. The largest MLA source trace uses this same format to fit
the source repository's limit. Reassembled capture bytes and provenance hashes
stay unchanged. After recapturing MLA, run
`node scripts/chunk-trace-assets.mjs static/traces` before publishing its source.

The diagrams can show a window of image/text positions, while attention inspectors
retain the full sequence and full denominator. Cropped probabilities are never
renormalized. Final output probabilities use all vocabulary or classifier logits.

DeepSeek executes pinned upstream modules, streaming only the pretrained experts
selected by the actual router. This changes weight residency, not computation:
all 27 blocks, both shared experts, final normalization and the complete LM head
execute. Its top-six routing weights are **not renormalized**. The shared expert
contribution is added separately. MLA expands content K/V for inspection; an
optimized inference engine can absorb those projections and cache the 512D latent
plus the 64D rotary key. The UI's cache comparison describes those dimensions,
not total model memory. Unassigned tokenizer IDs retain their checkpoint logits.

Zamba2 executes the official modules with all six layer-specific adapters. Its
attention scale is 1/√64 despite 128D Q/K heads. Transformer output enters the
Mamba input normalization; the block residual adds the Mamba output to the original
block input. For CPU memory use, the mathematically equivalent SSD execution chunk
size is 16 instead of 256. The inspector's state is explicitly a **reconstructed
sequential state**, calculated from recorded inputs and verified against the
official chunked scan. All 64 heads and 128 state coordinates are saved for the
first four of 64 channels; other block activations are full tensors.

SmolVLM receives a supported 256×256 image resolution without crops. All 256 image
patches traverse all 12 vision blocks. Pixel shuffle groups 4×4 patches, and the
learned projector produces 16 image tokens inserted into the actual language
sequence. All 30 language blocks execute. The image used for these examples is
generated locally by the earlier image capture script.

## Reproduction

Use Python 3.12 and `scripts/requirements-traces.txt`. Keep model caches outside
the repository. T5, DeiT and SmolVLM scripts expect their pinned checkpoints in
the supplied Hugging Face cache (the revisions are constants at the top of each
script). MLA and Zamba2 fetch bounded public checkpoint ranges and record a hash
for every tensor used.

```sh
python scripts/full_mla_capture.py --help
python scripts/capture-t5-full.py --cache-dir /tmp/moe-hf-cache
python scripts/capture-vit-full.py --cache-dir /tmp/moe-hf-cache
python scripts/capture-vlm-full.py --cache-dir /tmp/moe-hf-cache
python scripts/capture-hybrid-full.py --help
npm test
npm run validate:traces
npm run build
```

Tests verify capture provenance, all layer/head counts, full attention dot products
and softmax denominators, causal masks, attention×V, layer-to-layer continuity,
residuals, FFNs, expert routing/shared sums, image projection and merging, selective
state recurrence, and actual output logits. Captures also compare observed and
unobserved execution. DeepSeek and Zamba2 full-model logits match bit for bit;
floating-point reconstruction tolerances are disclosed in each trace.

Earlier prefix captures under `static/architectures/` are retained as reproducible
historical fixtures; the tabs use the complete captures under `static/traces/`.

Architecture references: [DeepSeek-V2](https://arxiv.org/abs/2405.04434),
[Zamba2](https://arxiv.org/abs/2411.15242), [T5](https://arxiv.org/abs/1910.10683),
[ViT](https://arxiv.org/abs/2010.11929), and the pinned model implementations.
Checkpoints retain their respective licenses; complete weights are not distributed.
