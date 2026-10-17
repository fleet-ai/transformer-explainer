#!/usr/bin/env python3
"""Record actual T5 encoder, causal decoder and cross-attention forward passes.

python scripts/capture-seq2seq.py --cache-dir /tmp/moe-hf-cache
Only compact activations and probabilities are published; model weights stay in
the external cache. This is a complete pretrained-model inference, not a toy
reimplementation. All captured arrays come from the unmodified eval forward.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

MODEL = "google-t5/t5-small"
REVISION = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
WEIGHT_SHA = "bd944e5f1b3ad9b70dd9d00010a517059e19265671076b8b0a4a58d9491842bc"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor(name, value, *, rows=None, columns=None, selection=None, shape=None, role=None):
    item = {"name": name, "shape": list(shape or value.shape), "values": value.tolist()}
    if role:
        item["role"] = role
        if role == "attention":
            item["head"] = 0
    if rows is not None:
        item["rowLabels"] = rows
    if columns is not None:
        item["columnLabels"] = columns
    if selection:
        item["slice"] = selection
    return item


def capture(cache_dir):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=cache_dir,
                                     allow_patterns=["*.json", "*.model", "model.safetensors"]))
    assert digest(snapshot / "model.safetensors") == WEIGHT_SHA
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(snapshot, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation="eager").eval()
    prompts = ["translate English to German: The house is wonderful.",
               "translate English to French: The cat is sleeping."]
    examples = []
    checks = {"attentionRowSumMaxError": 0., "decoderFutureAttentionMax": 0.,
              "crossAttentionOutputMaxError": 0., "hookedVsUnhookedLogitMaxError": 0.}
    for example_index, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        generated = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        decoder_ids = generated[:, :-1]
        baseline = model(**inputs, decoder_input_ids=decoder_ids, output_attentions=True,
                         output_hidden_states=True, use_cache=False, return_dict=True)
        source_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        decoder_tokens = tokenizer.convert_ids_to_tokens(decoder_ids[0])
        observed = {}
        hooks = []
        for i in range(model.config.num_layers):
            for stack_name, stack in [("encoder", model.encoder), ("decoder", model.decoder)]:
                ff = stack.block[i].layer[-1].DenseReluDense
                def hook(_, args, output, key=f"{stack_name}-{i}"):
                    observed[key] = (args[0].detach().clone(), output.detach().clone())
                hooks.append(ff.register_forward_hook(hook))
            attention = model.decoder.block[i].layer[1].EncDecAttention
            for attr in ("q", "k", "v", "o"):
                def hook(_, args, output, key=f"cross-{i}-{attr}"):
                    observed[key] = (args[0].detach().clone(), output.detach().clone())
                hooks.append(getattr(attention, attr).register_forward_hook(hook))
        traced = model(**inputs, decoder_input_ids=decoder_ids, output_attentions=True,
                       output_hidden_states=True, use_cache=False, return_dict=True)
        for hook in hooks:
            hook.remove()
        checks["hookedVsUnhookedLogitMaxError"] = max(checks["hookedVsUnhookedLogitMaxError"],
                                                    (baseline.logits-traced.logits).abs().max().item())
        layers = []
        for i in range(model.config.num_layers):
            enc, dec, cross = traced.encoder_attentions[i][0, 0], traced.decoder_attentions[i][0, 0], traced.cross_attentions[i][0, 0]
            for attention in (enc, dec, cross):
                checks["attentionRowSumMaxError"] = max(checks["attentionRowSumMaxError"],
                    (attention.sum(-1)-1).abs().max().item())
            checks["decoderFutureAttentionMax"] = max(checks["decoderFutureAttentionMax"], dec.triu(1).abs().max().item())
            v = observed[f"cross-{i}-v"][1][0].reshape(len(source_tokens), model.config.num_heads, model.config.d_kv).transpose(0, 1)
            expected = torch.matmul(traced.cross_attentions[i][0], v).transpose(0, 1).reshape(len(decoder_tokens), -1)
            actual = observed[f"cross-{i}-o"][0][0]
            error = (expected-actual).abs().max().item()
            checks["crossAttentionOutputMaxError"] = max(checks["crossAttentionOutputMaxError"], error)
            operations = [
                {"id": "embedding", "label": "Token embeddings", "kind": "embedding",
                 "description": "Source and decoder tokens use the same learned 512D embedding table. T5 supplies position information through relative attention biases.",
                 "formula": "xᵢ = E[tokenᵢ]", "tensors": [
                    tensor("Source embeddings", traced.encoder_hidden_states[0][0, :, :4], rows=source_tokens, columns=[str(n) for n in range(4)], selection="All source tokens · coordinates 0–3", shape=[len(source_tokens),512]),
                    tensor("Decoder embeddings", traced.decoder_hidden_states[0][0, :, :4], rows=decoder_tokens, columns=[str(n) for n in range(4)], selection="All decoder input tokens · coordinates 0–3", shape=[len(decoder_tokens),512])]},
                {"id":"encoder-self", "label":"Encoder self-attention", "kind":"attention",
                 "description":"Every source token can attend to every source token. The matrix is a full recorded head, including relative-position bias.",
                 "formula":"Aenc = softmax(QKᵀ + relative position bias)",
                 "tensors":[tensor("Encoder attention · head 1", enc, rows=source_tokens, columns=source_tokens, role="attention")],
                 "metrics":[{"label":"Heads", "value":model.config.num_heads},{"label":"Head width", "value":model.config.d_kv}]},
                {"id":"decoder-self", "label":"Causal decoder attention", "kind":"attention",
                 "description":"The decoder sees the generated prefix. Future positions are masked to zero. <pad> is T5’s decoder start token.",
                 "formula":"Adec = softmax(QKᵀ + relative bias + causal mask)",
                 "tensors":[tensor("Decoder attention · head 1", dec, rows=decoder_tokens, columns=decoder_tokens, role="attention")]},
                {"id":"cross-attention", "label":"Encoder → decoder attention", "kind":"attention",
                 "description":"Queries come from the decoder; keys and values come from the encoder’s final output. Each row shows which source tokens a decoder token reads.",
                 "formula":"Across = softmax(Qdecoder Kencoderᵀ); context = Across Vencoder",
                 "tensors":[tensor("Cross-attention · head 1", cross, rows=decoder_tokens, columns=source_tokens, role="attention"),
                    tensor("Head 1 values", v[0, :, :4], rows=source_tokens, columns=[str(n) for n in range(4)], selection="Head 1 · coordinates 0–3", shape=[len(source_tokens),64]),
                    tensor("Recorded context", actual[:, :4], rows=decoder_tokens, columns=[str(n) for n in range(4)], selection="Head 1 · coordinates 0–3", shape=[len(decoder_tokens),512])],
                 "metrics":[{"label":"A × V reconstruction error", "value":error}]},
                {"id":"feed-forward", "label":"Feed-forward + residual", "kind":"projection",
                 "description":"Each T5 block applies RMS normalization, a 512 → 2048 → 512 ReLU feed-forward network and a residual addition. These are recorded decoder FFN inputs and outputs.",
                 "formula":"FFN(x) = Wo ReLU(Wi x); hidden = residual + FFN(RMSNorm(hidden))",
                 "tensors":[tensor("Normalized FFN input", observed[f"decoder-{i}"][0][0,:,:4], rows=decoder_tokens, selection="Coordinates 0–3", shape=[len(decoder_tokens),512]),
                    tensor("FFN output", observed[f"decoder-{i}"][1][0,:,:4], rows=decoder_tokens, selection="Coordinates 0–3", shape=[len(decoder_tokens),512])]},
            ]
            operations.insert(2, {"id":"encoder-memory", "label":"Final encoder memory", "kind":"projection",
                "description":"All six encoder blocks finish before cross-attention reads their final normalized states. This same source memory feeds every decoder block.",
                "formula":"memory = RMSNorm(Encoder₆(…Encoder₁(Esource)))",
                "tensors":[tensor("Final encoder memory",traced.encoder_last_hidden_state[0,:,:4],rows=source_tokens,selection="All source tokens · coordinates 0–3",shape=[len(source_tokens),512])]})
            operations.insert(3, {"id":"decoder-input", "label":"Decoder prefix embeddings", "kind":"embedding",
                "description":"The actual generated prefix, shifted right by the T5 start token, is fed into the decoder. The first decoder token is <pad>.",
                "formula":"decoder input = E[<pad>, generated tokens before this step]",
                "tensors":[tensor("Decoder input embeddings",traced.decoder_hidden_states[0][0,:,:4],rows=decoder_tokens,selection="All decoder input tokens · coordinates 0–3",shape=[len(decoder_tokens),512])]})
            probs = torch.softmax(traced.logits[0, -1].double(), -1)
            p, ids = probs.topk(8)
            predictions = [{"label": tokenizer.decode([token_id]) or tokenizer.convert_ids_to_tokens(token_id), "probability":prob} for token_id, prob in zip(ids.tolist(), p.tolist())]
            operations.append({"id":"output", "label":"Vocabulary probabilities", "kind":"output",
                "description":"The final decoder state projects through the shared embedding table. These are the next-token probabilities after the full recorded output prefix (normally EOS).",
                "formula":"p(next token) = softmax(E × hidden / √512)",
                "tensors":[tensor("Top next-token probabilities", p, rows=[tokenizer.convert_ids_to_tokens(int(n)) for n in ids], selection="Top 8 of 32,128 vocabulary entries; probabilities are not renormalized", role="probabilities")],
                "metrics":[{"label":"Vocabulary", "value":model.config.vocab_size}]})
            layers.append({"index":i,"label":f"Encoder + decoder block {i+1}","operations":operations})
        examples.append({"id":f"translation-{example_index+1}","label":"English → German" if example_index == 0 else "English → French",
            "input":{"text":prompt},"tokens":source_tokens,"decoderTokens":decoder_tokens,
            "output":{"text":tokenizer.decode(generated[0], skip_special_tokens=True),"predictions":predictions},
            "generatedTokenIds":generated[0].tolist(),"layers":layers})
    assert checks["hookedVsUnhookedLogitMaxError"] == 0
    assert checks["decoderFutureAttentionMax"] == 0
    assert checks["attentionRowSumMaxError"] < 2e-6
    assert checks["crossAttentionOutputMaxError"] < 2e-5
    document = {"schemaVersion":1,"architecture":"encoder-decoder","title":"Encoder–decoder transformer",
        "provenance":{"modelId":MODEL,"revision":REVISION,"weightsSha256":WEIGHT_SHA,
            "weightFile":"model.safetensors","configSha256":digest(snapshot/"config.json"),
            "captureScript":"scripts/capture-seq2seq.py","captureScriptSha256":digest(__file__),
            "libraryVersions":{name:importlib.metadata.version(name) for name in ("torch","transformers","huggingface-hub")},
            "capturedAt":datetime.now(timezone.utc).isoformat(),"dtype":"float32","device":"cpu","trained":True,
            "scope":"Full pretrained T5-small greedy generation, then one full teacher-forced forward pass of the actual generated prefix. All six encoder/decoder blocks; attention head 1; vector slices explicitly labeled.",
            "modelUrl":f"https://huggingface.co/{MODEL}/tree/{REVISION}"},
        "config":{"encoderLayers":6,"decoderLayers":6,"hiddenSize":512,"intermediateSize":2048,"attentionHeads":8,"headSize":64,"activation":"ReLU","positions":"relative attention bias","attentionMultiplier":1},
        "verification":[{"name":name,"maxAbsoluteError":error,"tolerance": 0 if name in ("decoderFutureAttentionMax","hookedVsUnhookedLogitMaxError") else 2e-5} for name,error in checks.items()],"validation":checks,"examples":examples}
    output = ROOT/"static/architectures/encoder-decoder.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(document,ensure_ascii=False,separators=(",",":"))+"\n")
    print(json.dumps({"output":str(output),"bytes":output.stat().st_size,"translations":[e["output"]["text"] for e in examples],"validation":checks}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir",default="/tmp/moe-hf-cache")
    capture(parser.parse_args().cache_dir)
