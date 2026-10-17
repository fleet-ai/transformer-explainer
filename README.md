# Transformer Explainer · Architecture Explorer

[Live explainer](https://moe-trace-explainer.foamygenet33.chatgpt.site/) ·
[Fleet fork](https://github.com/fleet-ai/transformer-explainer)

The original [Polo Club Transformer Explainer](https://github.com/poloclub/transformer-explainer)
interface, with seven architecture tabs beneath the title. The upstream
base commit is `bfe50afba10b9b560b84143ee1107d977defa74f`.

- **Dense** (`/dense/`) restores the original GPT-2 diagram from that commit.
  Its five bundled examples contain recorded attention tensors and vocabulary
  logits. Custom generation uses the original ONNX model in the browser; its
  weights (626 MiB) are downloaded and cached from the upstream project's public
  model files on the first Generate. Recorded examples need no weight download.
  Projection illustrations are identified as schematic where values were not
  captured by the original model export.
- **MoE** (`/`) retains the Granite trace explorer and local inference service.
- **MLA** (`/architectures/mla/`) records the complete pretrained DeepSeek-V2-Lite:
  all 27 blocks, 16 attention heads, latent compression, decoupled RoPE, expanded
  content K/V, dense first block, routed and shared experts, final normalization,
  and all 102,400 vocabulary logits.
- **Encoder–decoder** (`/architectures/encoder-decoder/`) shows T5-small source
  attention, causal decoder attention and cross-attention for recorded translations.
- **Mamba hybrid** (`/architectures/hybrid/`) records all 38 Zamba2-1.2B blocks,
  including all six shared Transformer calls and their distinct adapters, the
  64-head selective-state computation in every block, and all 32,000 LM logits.
- **Vision–language** (`/architectures/vision-language/`) shows SmolVLM-256M's
  complete 12-block image encoder, pixel shuffle/projector and 30-block causal
  language decoder, with every attention head and the actual generated completion.
- **ViT** (`/architectures/vit/`) shows DeiT-tiny's image patches, class token,
  bidirectional attention and recorded ImageNet classification probabilities.

The five new views replay full pretrained-model captures in the original diagram
components. They do not run arbitrary inputs in the browser. Each inspector
identifies its checkpoint, tensor shape and displayed slice; trace downloads
expose the saved numbers and binary tensor manifests. See
[architecture capture notes](docs/architectures.md) for reproduction and limitations.

Each architecture owns its model state and inference runtime. The navigation
loads the corresponding application so GPT-2 tensors, tokenization, and layer
counts cannot be mixed with Granite's. All seven views ship in the same static site.

The recorded applications reuse the original `Embedding`, `QKV`, `Attention`,
`Sankey`, `SubsequentBlocks`, `LinearSoftmax`, and `BlockTransition` components,
along with the original input, temperature, and sampling controls. The visual
replacement for MoE is inside `Mlp.svelte`: router → expert bank → weighted sum.
Other architectures replace the stages their actual computation changes: MLA
compression, encoder cross-attention, selective-state updates, and image embedding.
Layout sizing accommodates those stages. No article or blog is rendered.

The diagram uses **real recorded forward passes** of
[IBM Granite 3.1 1B-A400M](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-base):
24 layers, 32 experts per layer, and 8 selected experts per token. The Examples
dropdown contains three saved prompts covering language, code, and arithmetic.
The editable prompt bar can also run the same model locally and display a newly
captured trace, including every attention head and the expert arithmetic.

The static trace bundle is approximately **12.61 MB** (12,613,955 bytes). It includes
attention scores and weights, router decisions, expert output previews, hidden-state
previews, the tokenizer vocabulary, and all 49,152 final-position vocabulary logits
for each prompt. Viewing the saved examples needs no inference server, API key,
GPU, or model-weight download. Custom prompts require the local service below.

## Develop and verify

Use Node.js 22.12+ or a supported newer release and npm 10+.

```sh
npm ci
npm run dev -- --host 127.0.0.1 --port 5178
```

The root development server runs the five architecture views and MoE at `http://127.0.0.1:5178/`. To develop
Dense separately, run `npm --prefix dense run dev -- --host 127.0.0.1 --port 5179`
and open `http://127.0.0.1:5179/dense/`. Use the combined preview below to test
the architecture tabs on one origin.

Automated verification and production build:

```sh
npm test
npm run validate:traces
npm run build
npm run preview -- --host 127.0.0.1
```

`npm run build` builds both applications and places the dense build in
`build/dense/`; the five recorded views are prerendered under `build/architectures/`. The preview serves this combined `build/` directory at
`http://127.0.0.1:4173/` (MoE) and `http://127.0.0.1:4173/dense/` (Dense).
Add `--port 4178` to use the local inference service's preauthorized preview
origin, or add `--allow-origin http://127.0.0.1:4173` when starting that service.
`build/` is the deployable static site. To deploy under a subpath, set `BASE_PATH`
when building, for example `BASE_PATH=/moe-explainer npm run build`.

**Known type-check limitation:** `npm run check` currently fails on legacy
TypeScript diagnostics in the retained upstream Svelte components and helpers.
It is not a passing verification gate. The production build, automated tests, and
trace validator are the current verification gates; CI runs those three commands.

## MoE custom prompts with real inference

The hosted page connects to a service on the **same computer as the browser** at
`http://127.0.0.1:8766`. It runs the pinned Granite checkpoint on CPU and keeps it
loaded between requests. It does not send prompts to a third-party inference API.
The service must remain running; a browser on another device needs its own service.

After installing the Python dependencies and downloading the checkpoint using
the reproduction commands below, start the service from this repository:

```sh
.venv/bin/python scripts/serve_traces.py --cache-dir /tmp/moe-hf-cache
```

On the development Mac, the existing environment/cache can be reused:

```sh
/private/tmp/moe-traces-venv/bin/python scripts/serve_traces.py --cache-dir /private/tmp/moe-hf-cache
```

Run `npm run build` once to make the diagram available at
`http://127.0.0.1:8766/` through this same service. This local app works directly
with Granite even when the hosted page's browser does not allow local-network
connections. The hosted page also links to it from **setup** or a connection error.
The service's `--web-root` option can select a different built app directory.

Enter text and click **Generate**. The service records a new forward pass and
checks it against an uninstrumented pass before exposing the trace. It retains
the latest 12 traces for that process under `/private/tmp/moe-live-traces`;
download a trace to keep it. Each prompt is limited to 32 tokenizer tokens and
4,096 characters (8 KB of UTF-8), and requests execute serially. Float32 model
weights alone need about 5 GiB RAM, with additional memory for captures.

The listener binds only to loopback. Browser submissions require the exact
explainer origin, a JSON request and the app's request header. Local preview
origins on ports 4178 and 5178 are allowed; another development origin can be
added explicitly with `--allow-origin`. Some browsers ask for local-network
access when Generate first connects. Stop a foreground service with Ctrl-C.

Service tests (no model download) run with:

```sh
python3 -m unittest discover -s tests -p 'test_live_service.py'
```

## MoE controls and recorded inference

- **Examples** selects one of the three saved prompts. **Generate** runs the edited
  prompt through the local model and replaces the diagram only after its real trace
  succeeds. While editing, the diagram keeps the last completed trace. The next-token
  prediction appears directly after the prompt; the next Generate includes that
  token exactly once. Selecting an example resets its inspection state.
- **Temperature, top-k, and top-p** are the original output-distribution controls.
  They recompute probabilities from all 49,152 recorded vocabulary logits, apply
  the selected filter, and renormalize over the retained vocabulary. This requires
  no new model inference. The display shows a subset of candidates while using
  the full vocabulary for normalization.
- **Expert routing remains top-8.** Vocabulary top-k/top-p controls do not change
  expert selection, attention, or hidden states. Changing the model's expert-routing
  configuration would require a new recording.
- The original head and block controls inspect the recorded attention at each
  selected head/layer. Token and expert selection inside the MoE stage inspect its
  recorded routes and outputs.
- **Inspect RoPE** shows four coordinates as two adjacent rotation pairs, with
  recorded before/after values, each pair's cosine and sine, and four substituted
  equations. The coordinate picker covers 0–31 in two rows, with each partner fixed
  at the selected coordinate plus 32. Both displayed pairs stay in range. The inspector follows the active Granite trace and its
  64D heads. Each pair simultaneously shows its **Q/K coordinate pair** as
  blue/purple before/after arrows on the same labeled radius and **cos / sin**
  rotation factors on a unit circle. Dotted lines project the selected coordinate
  onto its axis. Both diagrams update together, with the angle shown modulo 360°.

## Inspect expert arithmetic

Click an expert to open its recorded **Up projection → Gate projection → SiLU × up → Down projection** operations. Each projection displays a 4×4 parameter slice, input/output values, and a selectable output coordinate. The dot-product equation includes the signed contribution from all remaining input coordinates (1,020 or 508), so the preview does not pretend a four-term sum is the full multiplication. Operations are loaded per prompt when an expert is first inspected.

Click any **Σ** node to see all eight routing weights multiplied by the corresponding expert output coordinate, followed by their sum. Choose among the first eight output coordinates and inspect the separate ×0.22 residual contribution. An expert link in the sum opens its multiplication view. Skipped experts have no invented calculations.

## Reproduce the traces

`static/traces/granite-moe.json` records the immutable model revision, weight-shard
and capture-script SHA-256 hashes, packages, device, numeric precision, prompts,
and validation results. The offline capture executes the pretrained checkpoint in
eval mode and observes the implementation with forward hooks.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r scripts/requirements-traces.txt
.venv/bin/python scripts/capture_moe_traces.py --cache-dir /tmp/moe-hf-cache
npm run validate:traces
npm test
```

The checkpoint download is approximately 2.67 GB; CPU float32 inference requires
several GB of additional RAM. Validation alone needs no ML packages:

```sh
python3 scripts/capture_moe_traces.py --validate static/traces/granite-moe.json
```

After recapturing, copy the exact capture script and requirements into `static/traces/`
so the downloadable files match the provenance hash.

## Reading the numbers

- The model selects the top 8 router logits and softmaxes those selected logits.
  These normalized weights are the gates actually used by the experts. Full-router
  probabilities are separately derived from all 32 captured logits for comparison.
- Expert previews show the first 8 of 1,024 coordinates. Norms and reconstruction
  checks use the full 1,024-dimensional vectors.
- The displayed weighted sum is the raw MoE output. Granite scales it by its
  residual multiplier, 0.22, before adding the residual stream.
- Expert IDs are local to a layer. Expert 7 in one layer is a different network
  from expert 7 in another; these prompts do not establish topic specializations.
- Attention scores, masked scores, and causal softmax weights are recorded for all
  24 layers and 16 query heads. Granite uses 8 key/value heads, RoPE, RMSNorm, and
  a configured attention multiplier of 1/64. The adapter derives pre-scaling dot
  products by dividing recorded scores by that multiplier.
- Embedding previews include the model's ×12 multiplier. Attention-output previews
  precede residual scaling; final hidden previews follow the final RMSNorm.
- Captured activation previews are supplied to the retained vector components.
  Vectors without captured coordinates are neutral schematics, rather than random
  activation data. The Q/K/V structure is retained without inventing its activations.
- Next-token logits describe the position after the complete prompt and all model
  layers. Inspecting an intermediate layer does not create a new prediction.
- Live next-token candidate probabilities are derived with float64 softmax over
  the recorded logits. This avoids float32 denominator accumulation error across
  the vocabulary without changing the model's float32 forward pass. Trace
  validation retains the float32 discrepancy as a diagnostic and checks the
  serialized display probabilities against the saved logits.

## Automated coverage

The tests independently recalculate router softmax, selected expert weights,
expert loads, weighted output coordinates, attention normalization, and causal
masks from the recordings. Deliberately corrupted values exercise validator
failures. Adapter tests cover temperature and top-k/top-p filtering over complete
probability distributions, original attention-slot mappings, and token/model
alignment when a recording is selected.

The capture also checks full-vector reconstruction before serialization and
compares hooked versus unhooked model logits. See
[the verification inventory](docs/qa-inventory.md) for browser checks to perform;
that inventory is a checklist, not a blanket claim that each interaction passed.

## Attribution and license

Based on Transformer Explainer by Aeree Cho, Grace C. Kim, Alexander Karpekov,
Alec Helbling, Jay Wang, Seongmin Lee, Benjamin Hoover, and Polo Chau at Georgia Tech.
The upstream MIT [LICENSE](LICENSE) is retained. The original Svelte components,
D3/SVG connections, transitions, and controls remain the active interface, with
recorded Granite tensors adapted to their data contracts and the MLP stage replaced.
The upstream GPT-2 weight assets are omitted from this static build.

IBM's Granite model is Apache-2.0 licensed. The expert-operation recordings include small parameter slices; the complete checkpoint is not included. The Apache-2.0 license and attribution are in `static/traces/licenses/`.
This project is an independent educational adaptation, not an official Polo Club or
IBM release. See [upstream provenance](docs/upstream.md).
