# Original UI / MoE verification inventory

User story: open the original Transformer Explainer interface → choose a recorded
prompt → inspect attention and the replacement MoE stage → examine the final
next-token distribution.

The items below describe verification coverage and a browser checklist. An
unchecked browser item is not a claim that the interaction has been tested.

## Automated verification

- `npm test`: recorded routing and weighted-output contracts; causal attention and
  shape validation; capture provenance; adapter hydration and output-probability
  calculations, including temperature, top-k, and top-p.
- `npm run validate:traces`: validates the actual trace JSON, including alignment of
  tokens/layers/heads, expert choices/weights, attention, and preview norms.
- `npm run build`: compiles and prerenders the static application using the original
  UI components.
- `npm run check`: **known failing check**, with legacy original-component and
  helper TypeScript diagnostics. Do not report a clean full type-check.

## Browser checklist

- [ ] The original Embedding, QKV, Attention, Sankey, remaining-block, and output
      components form the continuous diagram. Only MLP has a new visual structure;
      sizing accommodates it. No hero, article, blog, or separate routing dashboard.
- [ ] The original Examples dropdown selects all three recorded prompts, including
      code/newline tokens, and the prompt display remains read-only.
- [ ] Switching short → long → short prompts updates all token rows, attention
      matrices, MoE paths, and predictions without stale geometry.
- [ ] Replay reloads the chosen recording and resets inspection; it neither runs
      new inference nor appends tokens.
- [ ] Original head/block navigation updates the real attention and expert routing,
      including first/last boundaries and the original block transitions.
- [ ] Original embedding/attention expansions and relevant in-place inspection
      controls remain usable within the same diagram.
- [ ] Selecting a MoE token highlights its actual eight expert routes. Displayed
      mixing weights match that token and layer.
- [ ] Selecting active experts shows their recorded outputs; inactive experts are
      labeled skipped and have no invented output coordinates.
- [ ] Original temperature, top-k, and top-p controls update output probabilities
      from the complete 49,152-token logits and renormalize retained candidates.
- [ ] Vocabulary filtering leaves expert routing fixed at top-8 and does not alter
      recorded attention/hidden states. Selecting an intermediate layer keeps the
      prediction tied to the complete prompt.
- [ ] Trace download links resolve. Loading failure and retry behave as displayed.
- [ ] Desktop layout, horizontal navigation on narrow screens, and resize behavior
      preserve the original continuous diagram and avoid clipping inspection content.
- [ ] Production loading uses static assets and the approximately 12.61 MB trace
      bundle; it makes no model-weight download or inference request.

## Latest local run

Production preview verified at 1280×720 and the full-width desktop diagram:
Examples switch among language, code, and math; Replay returns to block/head 1;
head and block transitions update recorded data; embedding, attention, and
probability expanders open; selected expert details show actual measurements;
temperature and vocabulary sampling change probabilities. For the language
recording, top-k=5 gives 69.17% for ` Par` at temperature 1 and 43.45% at
temperature 2. No production-page JavaScript errors were observed in these flows.
The viewport uses horizontal panning below the 1600px diagram minimum.

The automated run passed all 18 tests, both JavaScript and Python trace validation,
and the static production build. The known full type-check limitation remains.

## Expert and Σ operations

The added operation views were checked in the production preview. Clicking Σ
on token 4 and then token 0 updates the selected token and resets coordinate
selection. Coordinates 2 and 7 show their actual eight signed weighted terms.
Expert links open the corresponding recorded multiplications; up, gate, SiLU ×
up, and down steps are selectable, with row 2 updating its dot-product equation.
Switching from the language recording to arithmetic loads the correct sidecar.
Escape and the close button dismiss the inspector. No JavaScript errors were
observed in these production flows.

All 27 tests, both trace validators, and the production build pass. Svelte-check
reports no diagnostics for the new operation components or helpers; the existing
full-project legacy diagnostics remain. The new recordings cover all 5,568
selected expert evaluations and 16,704 projections.
