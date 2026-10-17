# Upstream provenance

Repository: https://github.com/poloclub/transformer-explainer

Base commit: bfe50afba10b9b560b84143ee1107d977defa74f

License: MIT; the original LICENSE is preserved.

Fleet fork: https://github.com/fleet-ai/transformer-explainer

The fork's `main` branch contains the Dense and MoE adaptation. The local `origin`
remote points to Fleet; `upstream` points to Polo Club.

The local repository retains upstream Git history and the `upstream` remote, with the
adaptation on `codex/moe-recorded-traces`. Source packaging excludes Git internals,
installed packages, model caches, and build artifacts. The deployment contains two
separate applications: Granite MoE at `/` and the restored GPT-2 diagram at
`/dense/`. A shared navigation component performs full-page loads, keeping their
model stores and inference runtimes separate.

The static archive includes recorded examples for both architectures, but no
complete model weights. Dense custom generation downloads the original GPT-2
ONNX parts from the upstream public site on demand, verifies each part against
its recorded SHA-256, and runs inference in the browser. MoE custom generation
uses the separately running local Granite service.

## Active UI reuse

`dense/` restores the upstream GPT-2 components and its five example recordings.
Its MLP, 12 layers, and 12 attention heads remain separate from Granite's MoE
architecture. Recorded examples and fresh browser inference supply attention
operations and vocabulary logits. Projection and activation illustrations whose
values were not exported are explicitly schematic; they do not imply captured
GPT-2 activations.

The root route imports the original `Embedding.svelte`, `QKV.svelte`,
`Attention.svelte`, `Sankey.svelte`, `SubsequentBlocks.svelte`,
`LinearSoftmax.svelte`, `BlockTransition.svelte`, and related original inspection
components. The original `Topbar`, `InputForm`, `Temperature`, and `Sampling`
components remain in use. This is direct component reuse, not a redrawn SVG
substitute for the original architecture.

`Mlp.svelte` is the visual replacement: its original stage now contains a learned
router, a 32-expert bank with eight selected experts, a weighted sum, and in-place
expert inspection. Its incoming/outgoing vector selectors and residual connection
seams remain compatible with the upstream Sankey. The original layout is resized
to fit this wider stage. No article or blog section is rendered.

## Data integration

`src/lib/moe/adapter.ts` translates the recorded model tensors into the original
store and attention-output contracts. Targeted integration changes bind the
original components to Granite metadata and captured previews, update relevant
model explanations, and keep missing activation values visibly schematic.

The MoE Examples dropdown selects three recorded prompts. The editable prompt
bar displays the sampled next token inline; Generate accepts that visible token
for an unchanged prompt or submits the edited text. The local service records a
fresh forward pass and verifies it against an uninstrumented pass before the
diagram changes. Editing keeps the last completed trace visible.

Temperature and top-k/top-p controls recompute the next-token distribution from
all 49,152 captured vocabulary logits. They do not change Granite's fixed top-8
expert routing. The approximately 12.61 MB preset bundle contains attention
scores/weights, router logits/gates, expert and hidden-state previews, vocabulary
text, and capture provenance. Live captures additionally provide matching expert
and attention-operation sidecars.

## Verification scope

The adaptation adds reproducible capture scripts, trace validators, mathematical
contract tests, and tests of the data adapter. The old scheduled upstream-merge
workflow is replaced with CI running tests, trace validation, and a production
build of both applications so upstream updates remain deliberate. The build
places Dense under `build/dense/` alongside the root MoE application; the entire
`build/` directory is the static deployment artifact.

Full `npm run check` still reports legacy TypeScript diagnostics from the retained
upstream components and helpers. It is not currently a passing gate; successful
build/test/validation results must not be described as a clean full type-check.
