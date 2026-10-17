# Recorded Granite MoE traces

`granite-moe.json` contains actual forward-pass measurements from the pretrained
[IBM Granite 3.1 1B-A400M base model](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-base/tree/408b6e90baab8cf24f4aa9f8e19703ffa0a53b29),
revision `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`. The model weights are
Apache-2.0 licensed. The expert-operation sidecars include small 4×4 excerpts of
those pretrained weights; the full model weights are not distributed here.

Captured on October 15, 2026 at 23:58:46 UTC using CPU, float32, Python 3.12.12,
PyTorch 2.6.0, and Transformers 4.51.3. The bundle includes the full package
versions, model-shard SHA-256 hashes, capture-script hash, and validation results.

The three prompts contain 29 tokens total. Every token is recorded at all 24
MoE layers: 696 token/layer pairs and 5,568 selected expert evaluations. Each
layer contains 32 independently parameterized experts and routes each token to 8.
The same forward passes also record all 16 attention heads at each layer:
1,152 complete attention matrices and 11,136 query rows across the three prompts.
The bundle also includes raw and masked attention scores, real Q/K/V previews,
all 49,152 final-token logits per example, and one shared decoded vocabulary.
Its JSON size is about 12.6 MB, or 4.5 MB when gzip-compressed.

| Example | Exact prompt | Tokens |
| --- | --- | ---: |
| Language | `The capital of France is` | 5 |
| Code | `def add(a, b):` followed by a newline and `    return a +` | 11 |
| Math | `Question: What is 7 + 5?` followed by a newline and `Answer:` | 13 |

## What the numbers mean

- `attention`: actual post-softmax attention weights with shape
  `[layer][head][query token][key token]`. All rows sum to one and all positions
  after the query token are exactly zero. Attention dropout is disabled in eval
  mode. Granite uses 16 query heads, 8 key/value heads, head dimension 64, RoPE,
  and its configured attention-logit multiplier of 1/64.
- `attentionScores`: scaled QK scores before masking, reconstructed using the
  actual observed Q/K projection outputs and that forward pass's RoPE inputs.
  `attentionMaskedScores` adds the actual causal mask. Their softmax is checked
  to match the model-returned attention exactly. Masked future entries contain
  the actual float32 minimum, about -3.4028e38; the UI can display that sentinel
  as negative infinity. These scores are never inferred from log probabilities.
- Per-layer `queryPreview`, `keyPreview`, and `valuePreview`: shape
  `[head][token][8 coordinates]`, with corresponding `queryNorms`, `keyNorms`,
  and `valueNorms` computed over all 64 head coordinates. Queries and keys are
  after RoPE. The 8 key/value heads are each repeated for their 2 query heads,
  matching the model's grouped-query attention.
- `embeddingPreview` / `embeddingNorms`: the first eight coordinates and full
  L2 norm of each token's embedding lookup multiplied by 12, before layer 1.
  No learned positional vector is added; Granite applies rotary positions to
  queries and keys inside attention.
- Per-layer `attentionInputPreview` / `attentionInputNorms`: the attention
  module's actual input after the layer's input RMSNorm.
- Per-layer `attentionOutputPreview` / `attentionOutputNorms`: the attention
  module's actual output after its output projection, before multiplication by
  0.22 and addition to the residual stream.
- `finalHiddenPreview` / `finalHiddenNorms`: the last hidden vectors after the
  model's final RMSNorm and before its language-model head. The head's resulting
  logits are divided by the model's configured factor of 6 before softmax.
- `routerLogits`: captured from the actual router. Granite selects the 8 largest
  logits and applies softmax to those 8 to obtain its actual `routingWeights`.
- `routerProbabilities`: softmax over all 32 captured logits, computed for the
  visualization. These full-distribution probabilities are distinct from the
  selected routing weights used to combine outputs.
- `selectedExperts`: zero-based expert IDs, sorted by router score. All selected
  output and weight arrays use this order. An expert ID refers only to its layer;
  expert 0 in layer 1 has different parameters from expert 0 in layer 2.
- `expertOutputPreview` and `combinedOutputPreview`: the first 8 of 1,024 actual
  coordinates. The complete 1,024-coordinate vectors are used for norms and
  reconstruction verification, then discarded to keep the website small.
- `combinedOutputPreview`: the weighted MoE sum before multiplication by the
  decoder's residual multiplier (0.22) and addition to the residual stream.
- `nextTokenCandidates`: the 10 highest probabilities from the complete
  49,152-token vocabulary. They are not renormalized to sum to one. A token may
  be whitespace or part of a word: the language example predicts ` Par`, while
  the math example's most likely next token is a space.
- `nextTokenLogits`: all 49,152 final-token model logits, including the model's
  inherent division by 6. Shared `vocabulary` maps every vocabulary ID to its
  decoded text, preserving whitespace. Together these allow temperature,
  top-k, and top-p demonstrations over the full saved vocabulary distribution.
  They do not provide inference for new prompts or subsequent generated tokens.

The data does not assign semantic specialties to experts, simulate routing,
change expert-routing top-k, or generate a continuation. Choosing a token or layer explores
the recorded forward pass.

## Reproduce the capture

From the repository root, create an isolated Python environment and install:

```sh
python3 -m venv .trace-venv
.trace-venv/bin/python -m pip install -r scripts/requirements-traces.txt
TOKENIZERS_PARALLELISM=false .trace-venv/bin/python scripts/capture_moe_traces.py \
  --cache-dir .hf-cache
```

The first run downloads about 2.67 GB of public model weights. Inference uses
float32 on CPU by default. The output is `static/traces/granite-moe.json`.
The exact script and requirements are also served in this directory for download.

Validate the existing artifact using only the Python standard library:

```sh
python3 scripts/capture_moe_traces.py --validate static/traces/granite-moe.json
```

The repository additionally validates it with:

```sh
node scripts/validate-traces.mjs
npm test
```

Forward hooks observe model outputs without modifying them. Each prompt is
rerun without hooks, and all vocabulary logits at every input position must
match exactly; the recorded maximum differences are all zero. Reconstructed
expert sums pass `torch.testing.assert_close` on every coordinate using
`atol=1e-5` and `rtol=2e-6`. Float32 summation order produces a largest absolute
difference of 0.0009765625 on a large-magnitude coordinate; the largest
error/tolerance ratio is 0.100616, below the required limit of 1. The maximum
reconstruction error in the eight displayed coordinates is 0.000002345.

The attention and full-vocabulary extensions were captured from the same pinned model and prompts.
All original MoE measurements and next-token probabilities are exactly unchanged
from the earlier MoE-only capture. The embedding interpretation is independently
checked against `embed_tokens(input_ids) * embedding_multiplier` during capture.

Numeric arrays are rounded to nine decimal places when serialized; full-vector
checks run before rounding. Reproducing on another device or software version
can change floating-point results slightly. The pinned source revision,
original capture environment, and artifact validation make those differences
inspectable.

The original candidate probabilities preserve PyTorch CPU float32 softmax.
Independent uninstrumented passes confirmed that the large vocabulary reduction
differs from stable float64 softmax by at most 0.000028343 in probability. The
bundle records the measured difference and full float32 probability sum for
each example. Browser sampling recomputes from the real full logits in float64,
so its unsampled probabilities can differ by this small precision effect.

## Inspecting multiplication inside an expert

The separately loaded `expert-operations-language.json`,
`expert-operations-code.json`, and `expert-operations-math.json` files record
all 5,568 selected expert evaluations. They are approximately 2.19 MB, 4.81 MB,
and 5.69 MB respectively. The main `granite-moe.json` bundle is unchanged.

Each expert shows its actual SwiGLU computation:

1. The post-attention RMSNorm input enters independent gate and up projections.
2. The model applies SiLU to the gate output.
3. It multiplies SiLU(gate) and up element by element.
4. A down projection maps the resulting 512-coordinate vector to 1,024 coordinates.
5. The router weight multiplies the expert output, and the selected experts' vectors
   are summed. This sum is before residual multiplication by 0.22.

For every gate, up, and down projection, the sidecar records a real 4×4 weight
tile, the first four actual output coordinates, and a signed
`omittedContribution` for each output coordinate. This term is independently
computed from **all** omitted coordinates using the actual full input and
pretrained weights in float64. Gate/up terms cover the remaining 1,020 input
coordinates; down terms cover the remaining 508 coordinates of the actual full
SiLU(gate) × up vector. The omitted term is never a norm or a residual invented
by subtracting the visible dot product from its answer.

Matrix rows are output coordinates and columns are input coordinates. Thus,
for each displayed output, the four visible input/weight products plus its
signed omitted-coordinate term reconstruct the actual projection output.
Float32 accumulation is checked with the scale-aware bound
`1e-5 + 1e-6 * sum(abs(all dot-product terms))`, including cancellation cases.
Every one of the 16,704 captured projections passes; the largest error/tolerance
ratio is below 0.381. The capture also verifies the actual SiLU module output,
the full elementwise product, and exact full-vector weighted sums.

Each sidecar includes independent provenance, its capture-script SHA-256, and
the SHA-256 of the unchanged main bundle. All router logits and all final-token
vocabulary logits are verified against that main recording. The downloadable
`capture_expert_operations.py` requires the adjacent `capture_moe_traces.py`
helper and the same pinned Python dependencies.

Reproduce or validate from the repository root:

```sh
TOKENIZERS_PARALLELISM=false .trace-venv/bin/python scripts/capture_expert_operations.py \
  --cache-dir .hf-cache
python3 scripts/capture_expert_operations.py --validate
node scripts/validate-expert-operations.mjs
```

## Inspecting head splitting, rotary positions, and shared K/V

The lazily loaded `attention-operations-language.json`,
`attention-operations-code.json`, and `attention-operations-math.json` files
contain complete 64-coordinate vectors, approximately 5.37 MB, 11.82 MB, and
13.97 MB respectively. The main bundle and expert-operation sidecars remain
unchanged.

The capture observes the actual pretrained projection outputs and the original
installed Granite implementation's `apply_rotary_pos_emb` and `repeat_kv`
executions. Transparent wrappers call those original functions, record their
actual inputs and outputs, and return their original tensors unchanged. The
post-RoPE vectors are captured from that execution, rather than produced solely
by an independent reconstruction.

- `queryBefore` and `queryAfter` have shape `[16 heads][tokens][64 coordinates]`.
  The pre-rotation vectors are verified to be the exact reshape of the actual
  1,024-coordinate query projection output.
- `keyBefore`, `keyAfter`, and `value` retain the 8 unique K/V heads, with shape
  `[8 heads][tokens][64 coordinates]`. Their pre-rotation vectors exactly match
  the actual 512-coordinate projection outputs.
- `positions` records the actual zero-based token positions. `rotary.cos` and
  `rotary.sin` contain 32 actual coefficients per position. The implementation's
  repeated second halves are verified identical, as are the coefficients used
  by every layer.
- RoPE pairs coordinate `i` with `i+32`, for `i=0..31`. For pair `(a,b)`, the
  actual outputs satisfy `a' = a*cos - b*sin` and `b' = b*cos + a*sin`. Position
  zero leaves all coordinates unchanged.
- Query head `h` uses unique key/value head `floor(h/2)`. The actual outputs of
  `repeat_kv` are verified to follow this mapping exactly. V is checked directly
  against the actual input to its repetition operation, proving it is unchanged
  by rotary position encoding.

Validation reconstructs all 534,528 recorded coordinate rotations and all
120,960 scaled QK scores, and compares every post-rotation/QKV preview to the
main recording. Original forward-pass attention probabilities, router logits,
and full vocabulary logits remain identical. Float64 reconstructions allow
small float32 accumulation differences using scale-aware tolerances; the actual
captured values are retained.

Each file records its own capture metadata, source SHA-256, and the unchanged
main-bundle SHA-256. The downloadable `capture_attention_operations.py` uses
the adjacent `capture_moe_traces.py` helper and the same pinned dependencies.

```sh
TOKENIZERS_PARALLELISM=false .trace-venv/bin/python scripts/capture_attention_operations.py \
  --cache-dir .hf-cache
python3 scripts/capture_attention_operations.py --validate
```

## Separate 128-coordinate RoPE example

`rope-example-128.json` is a separate, lazily loaded 49.9 KB recording from
[Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca),
pinned to revision `c1899de289a04d12100db370d81485cdf75e47ca`. It records block 0,
query head 0 and unique key/value head 0 for the five tokens in
`The capital of France is`. The Granite model diagram, main recording and all
existing sidecars retain their original values and SHA-256 hashes.

This model has 128 coordinates per head, 16 query heads, 8 key/value heads,
28 blocks, and `rope_theta=1000000` with no RoPE scaling. Its attention scores
use `1/sqrt(128)`. Qwen applies per-head `q_norm` and `k_norm` before RoPE;
the stored `queryBefore` and `keyBefore` values are the actual normalized
inputs at that boundary. All five Q/K/V arrays have shape `[tokens][128]`.
The stored actual `rotary.cos` and `rotary.sin` arrays have shape `[tokens][64]`,
with the implementation's repeated second halves verified identical. RoPE
pairs coordinate `i` with `i+64` for all `i=0..63`, so coordinates 0 through 127
can be inspected.

The capture wraps the original Transformers `apply_rotary_pos_emb` execution
and records its actual outputs, returning the original tensors to the model.
It verifies normalized Q/K outputs exactly match the observed RoPE inputs,
the V projection reshaped into heads exactly matches the unchanged V received
by attention, and actual grouped K/V repetition is exact. An unhooked second
execution matches every vocabulary logit at every prompt position exactly.
The independent validator reconstructs all 640 stored Q/K pairs; the largest
absolute rotation error is below `8.46e-6`, within the documented float32
tolerance. Actual position coefficients also match the configured theta.

The artifact includes capture time, package versions, CPU/float32 precision,
the immutable revision, verified weight and tokenizer checksums, configuration
checksum, installed model-implementation checksum, and its capture-script
checksum. The downloadable `capture_rope_example.py` is self-contained and uses
the same pinned dependencies in `requirements-traces.txt`. Model weights are
downloaded to the caller's cache and are not included in this example file.
Reproduction downloads about 1.5 GB on the first run.

```sh
TOKENIZERS_PARALLELISM=false .trace-venv/bin/python scripts/capture_rope_example.py \
  --cache-dir .hf-cache
python3 scripts/capture_rope_example.py --validate
```
