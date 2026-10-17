# Recorded architecture views

All visualized numbers come from forward computations with pretrained checkpoint
tensors. The site ships the JSON recordings, not the model weights. The five new
views replay recorded examples; Dense and Granite keep their existing generation
runtimes. Source code and JSON downloads identify every capture's scope.

| Tab | Checkpoint | Captured computation |
| --- | --- | --- |
| MLA | [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | Embedding and first trained MLA attention layer; no full-model prediction |
| Encoder–decoder | [T5-small](https://huggingface.co/google-t5/t5-small) | Full translation generation and the resulting source/decoder prefix through all six paired blocks |
| Mamba hybrid | [Zamba2-1.2B](https://huggingface.co/Zyphra/Zamba2-1.2B) | Trained prefix through the first hybrid layer; no full-model prediction |
| Vision–language | [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | Image encoding, image-token projection and causal language decoder |
| ViT | [DeiT-tiny-patch16-224](https://huggingface.co/facebook/deit-tiny-patch16-224) | Full image classification with 196 patch tokens and one class token |

The models illustrate different mechanisms; their results are not a controlled
comparison of model quality. The illustration used for vision inputs is generated
locally by the capture script. Classification and generated text are preserved
exactly, including incorrect or awkward predictions.

## Trace contract

`static/architectures/*.json` uses schema version 1. Each file includes a pinned
checkpoint revision, weight hashes, library versions, a reproducible capture script,
model configuration and named examples. Examples contain layer operations with
formulas and recorded tensors. A tensor's `shape` is its original shape; `slice`
identifies omitted coordinates/rows. Labels describe the displayed coordinates.
Attention crops are not renormalized. Top output probabilities retain their original
vocabulary/class denominator.

For MLA and Zamba2, byte-range downloads retrieve only the needed checkpoint tensors
and prompt embedding rows. Every fetched tensor is identified and hashed. The
capture runs the real pretrained computation for that stated portion; no randomly
initialized substitute or fictional next-token prediction is used.

The MLA view expands K/V to expose the math. An optimized decoder can absorb the
projection weights and cache the compressed latent plus the decoupled rotary key;
it need not store all expanded keys and values. The cache comparison concerns these
specific dimensions, not total model memory.

## Reproduction and verification

Use the Python environment from `scripts/requirements-traces.txt`; the capture
scripts identify any additional dependencies in their imports. Model caches and
weights stay outside the source tree. Each script exposes command-line help.

```sh
python scripts/capture-mla-traces.py --help
python scripts/capture-hybrid-traces.py --help
python scripts/capture-seq2seq.py --help
python scripts/capture-multimodal.py --model vit --cache-dir /tmp/moe-hf-cache
python scripts/capture-multimodal.py --model vision-language --cache-dir /tmp/moe-hf-cache
npm test
npm run build
```

The dataset tests check finite tensor values, probability ranges, complete attention
row sums, checkpoint provenance, and cross-attention reconstruction from saved
weights and values. Capture scripts also validate intermediate numerical operations
against the executed model. The source SHA in a trace identifies the capture code.

Architecture references: [DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434),
[Zamba2 report](https://arxiv.org/abs/2411.15242),
[T5 paper](https://arxiv.org/abs/1910.10683),
[Vision Transformer paper](https://arxiv.org/abs/2010.11929), and the pinned model
implementations and model cards linked above. Checkpoints retain their respective
model licenses; the repository does not redistribute their complete weights.
