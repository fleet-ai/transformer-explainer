#!/usr/bin/env python3
"""One loaded, pinned Granite model for genuine interactive trace requests.

The original capture scripts remain immutable provenance records. This engine
combines their hooks into one instrumented pass, followed by an uninstrumented
pass that must match every vocabulary logit exactly. Only validated results are
published. The caller is responsible for request authentication and retention.
"""
from __future__ import annotations

import argparse
import codecs
from datetime import datetime, timezone
from functools import wraps
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import re
import threading
import time

from capture_moe_traces import MODEL_ID, MODEL_REVISION, rounded, sha256_file
from capture_attention_operations import validate_sidecar as validate_attention
from capture_expert_operations import validate_sidecar as validate_experts

MAX_TOKENS = 32
MAX_PROMPT_BYTES = 8192
# Nine-decimal candidate rounding adds at most 0.5e-9, and nine-decimal
# logit rounding adds at most 0.25e-9: a softmax component has Jacobian
# L1 norm 2*p*(1-p) <= 0.5. The remainder covers float64 arithmetic.
VOCABULARY_PROBABILITY_TOLERANCE = 1e-9
# Rotary observers temporarily wrap module globals, so even separate engine
# instances in one process must not execute the model concurrently.
_CAPTURE_LOCK = threading.Lock()


def validate_prompt(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a non-empty prompt.")
    try:
        encoded = prompt.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("The prompt contains invalid Unicode.") from error
    if len(encoded) > MAX_PROMPT_BYTES:
        raise ValueError(f"Use a prompt shorter than {MAX_PROMPT_BYTES} UTF-8 bytes.")
    return prompt


def validate_trace_id(trace_id):
    if not isinstance(trace_id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}", trace_id):
        raise ValueError("Invalid trace ID.")
    return trace_id


def decode_token_pieces(tokenizer, input_ids, prompt):
    """Lossless labels for byte-level BPE, including characters split over IDs.

    UTF-8 bytes incomplete at a token boundary are buffered until their character
    is complete. A token can therefore have an empty display label; its real ID
    and tensor row are retained. This avoids inventing replacement characters.
    """
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    byte_decoder = {character: byte for byte, character in bytes_to_unicode().items()}
    added = tokenizer.get_added_vocab()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    pieces = []
    for raw_token in tokenizer.convert_ids_to_tokens(input_ids):
        token_bytes = (raw_token.encode("utf-8") if raw_token in added else
                       bytes(byte_decoder[character] for character in raw_token))
        pieces.append(decoder.decode(token_bytes))
    tail = decoder.decode(b"", final=True)
    if pieces:
        pieces[-1] += tail
    if "".join(pieces) != prompt:
        raise ValueError("The model tokenizer could not preserve this prompt exactly.")
    return pieces



def validate_vocabulary_candidates(example, vocabulary):
    """Independently reconstruct derived probabilities from saved real logits."""
    logits = example["nextTokenLogits"]
    assert len(logits) == len(vocabulary) > 0 and all(math.isfinite(value) for value in logits)
    candidates = example["nextTokenCandidates"]
    assert len(candidates) == min(10, len(logits))
    ids = [candidate["id"] for candidate in candidates]
    assert all(type(token_id) is int and 0 <= token_id < len(logits) for token_id in ids)
    assert len(set(ids)) == len(ids), "Vocabulary candidate IDs must be unique"
    assert all(math.isfinite(candidate["probability"]) and 0 <= candidate["probability"] <= 1
               for candidate in candidates)
    assert all(candidates[index]["probability"] >= candidates[index + 1]["probability"]
               for index in range(len(candidates) - 1))
    # Preserve any valid tie ordering at the top-k boundary.
    threshold = sorted(logits, reverse=True)[len(candidates) - 1]
    assert min(logits[token_id] for token_id in ids) >= threshold, "Candidates must be top vocabulary logits"
    assert math.fsum(candidate["probability"] for candidate in candidates) <= 1 + len(candidates) * 0.5e-9
    maximum = max(logits)
    denominator = math.fsum(math.exp(value - maximum) for value in logits)
    max_error = 0.0
    for candidate in candidates:
        assert candidate["text"] == vocabulary[candidate["id"]]
        expected = math.exp(logits[candidate["id"]] - maximum) / denominator
        error = abs(candidate["probability"] - expected)
        max_error = max(max_error, error)
        assert error <= VOCABULARY_PROBABILITY_TOLERANCE, f"Vocabulary probability error: {error}"
    return max_error


def validate_bundle(bundle: dict) -> dict:
    """Validate live tensors; allow 1e-6 float32 probability-mass error.

    A 32-way float32 reduction can differ from one by more than 2e-7 on
    arbitrary prompts. Router/attention values remain the actual float32
    tensors. Vocabulary probabilities are derived in float64 from captured
    logits and checked independently to their nine-decimal serialization bound.
    """
    assert bundle["schemaVersion"] == 1
    config = bundle["model"]
    assert len(bundle["vocabulary"]) == config["vocabSize"]
    assert all(isinstance(text, str) for text in bundle["vocabulary"])
    n_experts, top_k = config["numExperts"], config["topK"]
    token_layer_count = 0
    attention_row_count = 0
    max_preview_error = 0.0
    max_recorded_error = 0.0
    max_vocabulary_probability_error = 0.0
    for example in bundle["examples"]:
        assert example["validation"]["hookedVsUnhookedMaxLogitError"] == 0
        n_tokens = len(example["tokens"])
        assert "".join(token["text"] for token in example["tokens"]) == example["prompt"]
        assert len(example["layers"]) == config["numLayers"]
        assert [token["index"] for token in example["tokens"]] == list(range(n_tokens))
        def validate_previews(previews, norms):
            assert len(previews) == len(norms) == n_tokens
            for vector, norm in zip(previews, norms):
                assert len(vector) == 8 and all(math.isfinite(value) for value in vector)
                assert math.isfinite(norm) and norm >= 0
                assert math.sqrt(sum(value * value for value in vector)) <= norm + 1e-5

        validate_previews(example["embeddingPreview"], example["embeddingNorms"])
        validate_previews(example["finalHiddenPreview"], example["finalHiddenNorms"])
        assert len(example["attention"]) == config["numLayers"]
        assert len(example["attentionScores"]) == len(example["attentionMaskedScores"]) == config["numLayers"]
        for layer_index, heads in enumerate(example["attention"]):
            assert len(heads) == config["attentionHeads"]
            score_heads = example["attentionScores"][layer_index]
            masked_heads = example["attentionMaskedScores"][layer_index]
            assert len(score_heads) == len(masked_heads) == len(heads)
            for head_index, matrix in enumerate(heads):
                assert len(matrix) == n_tokens
                assert len(score_heads[head_index]) == len(masked_heads[head_index]) == n_tokens
                for query, row in enumerate(matrix):
                    attention_row_count += 1
                    scores = score_heads[head_index][query]
                    masked = masked_heads[head_index][query]
                    assert len(scores) == len(masked) == n_tokens
                    assert all(math.isfinite(value) for value in scores + masked)
                    assert len(row) == n_tokens
                    assert all(math.isfinite(value) and 0 <= value <= 1 for value in row)
                    assert math.isclose(sum(row), 1, abs_tol=1e-6)
                    assert all(value == 0 for value in row[query + 1:]), "Future-token attention must be masked"
                    assert masked[:query + 1] == scores[:query + 1]
                    assert all(value == config["attentionMaskValue"] for value in masked[query + 1:])
                    exponentials = [math.exp(value - max(masked)) for value in masked]
                    denominator = sum(exponentials)
                    assert all(abs(value - exp / denominator) < 2e-7
                               for value, exp in zip(row, exponentials))
        for layer_index, layer in enumerate(example["layers"]):
            assert layer["index"] == layer_index
            assert len(layer["tokens"]) == n_tokens
            validate_previews(layer["attentionInputPreview"], layer["attentionInputNorms"])
            validate_previews(layer["attentionOutputPreview"], layer["attentionOutputNorms"])
            for kind in ("query", "key", "value"):
                assert len(layer[f"{kind}Preview"]) == len(layer[f"{kind}Norms"]) == config["attentionHeads"]
                for head_previews, head_norms in zip(layer[f"{kind}Preview"], layer[f"{kind}Norms"]):
                    validate_previews(head_previews, head_norms)
            counts = [0] * n_experts
            for token in layer["tokens"]:
                token_layer_count += 1
                probabilities = token["routerProbabilities"]
                logits = token["routerLogits"]
                selected = token["selectedExperts"]
                weights = token["routingWeights"]
                assert len(probabilities) == len(logits) == n_experts
                assert len(selected) == len(set(selected)) == len(weights) == top_k
                assert all(math.isfinite(x) for x in logits + probabilities + weights)
                assert all(0 <= p <= 1 for p in probabilities + weights)
                assert math.isclose(sum(probabilities), 1, abs_tol=1e-6)
                assert math.isclose(sum(weights), 1, abs_tol=1e-6)
                assert selected == sorted(range(n_experts), key=lambda i: logits[i], reverse=True)[:top_k]
                softmax_denominator = sum(math.exp(logits[i] - max(logits)) for i in range(n_experts))
                expected_probabilities = [math.exp(x - max(logits)) / softmax_denominator for x in logits]
                assert all(abs(a-b) < 2e-7 for a, b in zip(probabilities, expected_probabilities))
                selected_mass = sum(probabilities[i] for i in selected)
                assert all(abs(weight - probabilities[expert] / selected_mass) < 2e-7
                           for weight, expert in zip(weights, selected))
                expert_preview = token["expertOutputPreview"]
                combined_preview = token["combinedOutputPreview"]
                assert len(expert_preview) == top_k
                assert len(combined_preview) == 8
                assert all(len(row) == 8 for row in expert_preview)
                for coordinate in range(8):
                    reconstructed = sum(weights[i] * expert_preview[i][coordinate] for i in range(top_k))
                    error = abs(reconstructed - combined_preview[coordinate])
                    max_preview_error = max(max_preview_error, error)
                    assert math.isclose(reconstructed, combined_preview[coordinate], rel_tol=2e-6, abs_tol=1e-5)
                assert len(token["expertOutputNorms"]) == len(token["weightedOutputNorms"]) == top_k
                assert all(math.isclose(out * weight, weighted, rel_tol=2e-6, abs_tol=1e-6)
                           for out, weight, weighted in zip(token["expertOutputNorms"], weights,
                                                            token["weightedOutputNorms"]))
                error = token["reconstructionMaxAbsError"]
                assert math.isfinite(error) and error >= 0
                assert token["reconstructionToleranceRatio"] <= 1
                assert error <= 1e-5 + 2e-6 * token["outputMaxAbs"]
                max_recorded_error = max(max_recorded_error, error)
                for expert in selected:
                    counts[expert] += 1
            assert layer["expertTokenCounts"] == counts
            assert sum(counts) == n_tokens * top_k
        max_vocabulary_probability_error = max(
            max_vocabulary_probability_error,
            validate_vocabulary_candidates(example, bundle["vocabulary"]),
        )
    return {"passed": True, "examples": len(bundle["examples"]),
            "probabilityMassTolerance": 1e-6,
            "maxRouterProbabilityMassError": max(abs(sum(token["routerProbabilities"]) - 1)
                                                for example in bundle["examples"]
                                                for layer in example["layers"] for token in layer["tokens"]),
            "tokenLayerPairs": token_layer_count,
            "attentionRows": attention_row_count,
            "maxSerializedPreviewReconstructionError": max_preview_error,
            "maxFullVectorReconstructionError": max_recorded_error,
            "vocabularyProbabilityTolerance": VOCABULARY_PROBABILITY_TOLERANCE,
            "maxVocabularySoftmaxFloat64ReconstructionError": max_vocabulary_probability_error,
            "checks": ["all input tokens and all routed layers present", "router softmax matches logits",
                       "selected experts match top-k logits", "selected weights sum to one",
                       "weighted expert previews reconstruct captured MoE output",
                       "full-vector reconstruction within atol=1e-5 plus rtol=2e-6", "expert counts match routing",
                       "hooked and uninstrumented passes have identical vocabulary logits",
                       "all attention heads present; rows sum to one and future positions are exactly zero",
                       "true scaled QK scores plus actual causal mask reproduce captured attention softmax",
                       "full vocabulary logits reproduce next-token candidates without top-10 renormalization",
                       "embedding, final hidden, and attention vectors have finite previews and full-vector norms",
                       "next-token candidates are sorted vocabulary probabilities"]}


class LiveTraceEngine:
    def __init__(self, cache_dir, threads=4):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not isinstance(threads, int) or not 1 <= threads <= 32:
            raise ValueError("threads must be between 1 and 32")
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        self.threads = threads
        self.snapshot = Path(snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir,
            local_files_only=True, allow_patterns=["*.json", "*.safetensors"],
        ))
        self.tokenizer = AutoTokenizer.from_pretrained(self.snapshot, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.snapshot, local_files_only=True, torch_dtype=torch.float32,
            low_cpu_mem_usage=True, attn_implementation="eager",
        ).eval()
        self.model.requires_grad_(False)
        if self.model.config.model_type != "granitemoe" or self.model.config.hidden_act != "silu":
            raise ValueError("The installed model does not match the Granite capture layout.")
        self.model_files = [{"name": path.name, "bytes": path.stat().st_size,
                             "sha256": sha256_file(path)}
                            for path in sorted(self.snapshot.glob("*.safetensors"))]
        # Compare cached weights to the previously independently captured bundle.
        reference_path = Path(__file__).resolve().parents[1] / "static/traces/granite-moe.json"
        if reference_path.exists():
            reference = json.loads(reference_path.read_text())
            if reference["model"]["revision"] != MODEL_REVISION or reference["capture"]["modelFiles"] != self.model_files:
                raise ValueError("Cached model weights do not match the pinned reference recording.")
        self.config_sha256 = sha256_file(self.snapshot / "config.json")
        self.packages = {name: metadata.version(name) for name in
                         ("torch", "transformers", "safetensors", "huggingface-hub", "tokenizers")}
        self.vocabulary = [self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                           for token_id in range(self.model.config.vocab_size)]
        self.model_metadata = self._model_metadata(torch)

    def _model_metadata(self, torch):
        config = self.model.config
        return {
            "id": MODEL_ID, "name": "Granite 3.1 1B-A400M", "revision": MODEL_REVISION,
            "url": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
            "license": "Apache-2.0", "numLayers": config.num_hidden_layers,
            "numExperts": config.num_local_experts, "topK": config.num_experts_per_tok,
            "hiddenSize": config.hidden_size, "intermediateSize": config.intermediate_size,
            "totalParameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "activeParametersApprox": 400_000_000, "residualMultiplier": config.residual_multiplier,
            "attentionHeads": config.num_attention_heads, "keyValueHeads": config.num_key_value_heads,
            "attentionHeadSize": config.hidden_size // config.num_attention_heads,
            "attentionMultiplier": config.attention_multiplier, "embeddingMultiplier": config.embedding_multiplier,
            "logitsScaling": config.logits_scaling, "ropeTheta": config.rope_theta,
            "vocabSize": config.vocab_size, "attentionMaskValue": torch.finfo(torch.float32).min,
        }

    def capture(self, prompt, trace_id, output_dir):
        validate_prompt(prompt)
        validate_trace_id(trace_id)
        with _CAPTURE_LOCK:
            return self._capture(prompt, trace_id, Path(output_dir))

    def _capture(self, prompt, trace_id, output_dir):
        import torch
        from transformers.models.granitemoe import modeling_granitemoe as implementation
        # Save originals before installing transparent observers. The main
        # attention arithmetic checks deliberately call these saved functions.
        apply_rotary_pos_emb = implementation.apply_rotary_pos_emb
        repeat_kv = implementation.repeat_kv
        tokenizer, model = self.tokenizer, self.model
        config = model.config
        top_k, n_experts, n_layers = config.num_experts_per_tok, config.num_local_experts, config.num_hidden_layers
        head_size = config.hidden_size // config.num_attention_heads
        group_size = config.num_attention_heads // config.num_key_value_heads
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"][0].tolist()
        if not 1 <= len(input_ids) <= MAX_TOKENS:
            raise ValueError(f"This prompt has {len(input_ids)} tokens; use at most {MAX_TOKENS} tokens.")
        labels = decode_token_pieces(tokenizer, input_ids, prompt)
        files = {"main": "main.json", "attention": f"attention-operations-{trace_id}.json",
                 "expert": f"expert-operations-{trace_id}.json"}
        if any((output_dir / filename).exists() for filename in files.values()):
            raise FileExistsError("This trace ID already has captured data.")
        start = time.monotonic()
        current_layers = {}
        router_cache = {}
        expert_cache = {}
        attention_cache = {}
        projection_cache = {}
        attention_score_cache = {}
        handles = []

        def projection_hook(index, kind):
            def hook(_module, _inputs, output):
                projection_cache.setdefault(index, {})[kind] = output.detach()
            return hook

        def attention_hook(index):
            def hook(module, _inputs, kwargs, output):
                # Inputs are after input_layernorm. Outputs are after o_proj, before
                # the decoder scales the branch and adds the residual.
                attention_input = kwargs["hidden_states"].detach().cpu()[0]
                attention_output = output[0].detach().cpu()[0]
                projections = projection_cache.pop(index)
                batch_size, length, _ = kwargs["hidden_states"].shape
                query = projections["query"].view(batch_size, length, module.num_heads, module.head_dim).transpose(1, 2)
                key = projections["key"].view(batch_size, length, module.num_key_value_heads, module.head_dim).transpose(1, 2)
                value = projections["value"].view(batch_size, length, module.num_key_value_heads, module.head_dim).transpose(1, 2)
                cos, sin = kwargs["position_embeddings"]
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
                key = repeat_kv(key, module.num_key_value_groups)
                value = repeat_kv(value, module.num_key_value_groups)
                # Recompute the exact attention arithmetic from observed projection
                # outputs and the actual RoPE/mask inputs. Never infer QK from log(p).
                scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling
                masked_scores = scores + kwargs["attention_mask"][:, :, :, :key.shape[-2]]
                reconstructed_attention = masked_scores.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
                torch.testing.assert_close(reconstructed_attention, output[1], rtol=0, atol=0)
                attention_score_cache[index] = {
                    "scores": rounded(scores.detach().cpu()[0].tolist()),
                    "masked": rounded(masked_scores.detach().cpu()[0].tolist()),
                }
                attention_cache[index] = {
                    "attentionInputPreview": rounded(attention_input[:, :8].tolist()),
                    "attentionInputNorms": rounded(attention_input.norm(dim=-1).tolist()),
                    "attentionOutputPreview": rounded(attention_output[:, :8].tolist()),
                    "attentionOutputNorms": rounded(attention_output.norm(dim=-1).tolist()),
                }
                for kind, tensor in (("query", query), ("key", key), ("value", value)):
                    head_vectors = tensor.detach().cpu()[0]
                    attention_cache[index][f"{kind}Preview"] = rounded(head_vectors[:, :, :8].tolist())
                    attention_cache[index][f"{kind}Norms"] = rounded(head_vectors.norm(dim=-1).tolist())
            return hook

        def router_hook(index):
            def hook(_module, inputs, output):
                sorted_indices, batch_indices, batch_gates, expert_sizes, logits = output
                router_cache[index] = {
                    "sorted_indices": sorted_indices.detach().cpu(),
                    "batch_indices": batch_indices.detach().cpu(),
                    "batch_gates": batch_gates.detach().cpu(),
                    "expert_sizes": expert_sizes,
                    "logits": logits.detach().cpu(),
                    "input": inputs[0].detach().cpu(),
                }
            return hook

        def expert_hook(index):
            def hook(_module, _inputs, output):
                # Actual expert outputs, before the model multiplies by routing gates.
                expert_cache[index] = output.detach().cpu()
            return hook

        def moe_hook(index):
            def hook(_module, _inputs, output):
                state = router_cache.pop(index)
                grouped_outputs = expert_cache.pop(index)
                actual_outputs, actual_logits = output
                actual_outputs = actual_outputs.detach().cpu().reshape(-1, config.hidden_size)
                torch.testing.assert_close(actual_logits.detach().cpu(), state["logits"], rtol=0, atol=0)
                logits = state["logits"]
                token_count = logits.shape[0]
                selected_logits, selected_experts = logits.topk(top_k, dim=-1)
                expected_weights = selected_logits.softmax(dim=-1)
                # Undo the model's grouping by expert, preserving its exact token/top-k ordering.
                flat_outputs = torch.empty_like(grouped_outputs)
                flat_outputs[state["sorted_indices"]] = grouped_outputs
                expert_outputs = flat_outputs.reshape(token_count, top_k, config.hidden_size)
                flat_weights = torch.empty_like(state["batch_gates"])
                flat_weights[state["sorted_indices"]] = state["batch_gates"]
                weights = flat_weights.reshape(token_count, top_k)
                torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
                weighted_outputs = expert_outputs * weights[:, :, None]
                reconstructed = weighted_outputs.sum(dim=1)
                errors = (actual_outputs - reconstructed).abs().amax(dim=-1)
                tolerance_ratios = ((actual_outputs - reconstructed).abs() /
                                    (1e-5 + 2e-6 * actual_outputs.abs())).amax(dim=-1)
                torch.testing.assert_close(reconstructed, actual_outputs, rtol=2e-6, atol=1e-5)
                probabilities = logits.softmax(dim=-1)
                expert_norms = expert_outputs.norm(dim=-1)
                weighted_norms = weighted_outputs.norm(dim=-1)
                tokens = []
                for token_index in range(token_count):
                    tokens.append({
                        "routerLogits": rounded(logits[token_index].tolist()),
                        "routerProbabilities": rounded(probabilities[token_index].tolist()),
                        "selectedExperts": selected_experts[token_index].tolist(),
                        "routingWeights": rounded(weights[token_index].tolist()),
                        "expertOutputNorms": rounded(expert_norms[token_index].tolist()),
                        "weightedOutputNorms": rounded(weighted_norms[token_index].tolist()),
                        "expertOutputPreview": rounded(expert_outputs[token_index, :, :8].tolist()),
                        "combinedOutputPreview": rounded(actual_outputs[token_index, :8].tolist()),
                        "inputNorm": round(state["input"][token_index].norm().item(), 9),
                        "outputNorm": round(actual_outputs[token_index].norm().item(), 9),
                        "outputMaxAbs": actual_outputs[token_index].abs().max().item(),
                        "reconstructionMaxAbsError": errors[token_index].item(),
                        "reconstructionToleranceRatio": tolerance_ratios[token_index].item(),
                    })
                current_layers[index] = {"index": index, "tokens": tokens,
                                         "expertTokenCounts": state["expert_sizes"],
                                         **attention_cache.pop(index)}
            return hook

        ops_cache = {}
        ops_layers = {}

        def ops_router_hook(index):
            def hook(_module, inputs, output):
                sorted_indices, batch_indices, gates, expert_sizes, logits = output
                ops_cache[index] = {"input": inputs[0].detach(), "sortedIndices": sorted_indices,
                                "batchIndices": batch_indices, "gates": gates, "logits": logits}
            return hook

        def ops_input_hook(index):
            def hook(module, inputs, output):
                ops_cache[index].update(linearInput=inputs[0].detach(), linearOutput=output.detach(),
                                    inputWeights=module.weight.detach())
            return hook

        def ops_activation_hook(index):
            def hook(_module, _inputs, output):
                ops_cache[index]["silu"] = output.detach()
            return hook

        def ops_output_hook(index):
            def hook(module, inputs, output):
                ops_cache[index].update(product=inputs[0].detach(), downOutput=output.detach(),
                                    outputWeights=module.weight.detach())
            return hook

        def project(weights, full_input, actual_output):
            # FP64 dot products over all omitted inputs, from the actual forward
            # pass. This signed tail is never obtained by subtracting the preview
            # from the output, and is never an activation or weight norm.
            rows = weights[:4].double()
            inputs = full_input.double()
            omitted = rows[:, 4:] @ inputs[4:]
            reconstructed = rows[:, :4] @ inputs[:4] + omitted
            actual = actual_output[:4].double()
            # Bound FP32 accumulation error by the sum of absolute dot-product
            # terms; this remains meaningful even when signed terms cancel.
            sum_absolute_terms = rows.abs() @ inputs.abs()
            tolerance = 1e-5 + 1e-6 * sum_absolute_terms
            error = (reconstructed - actual).abs()
            assert torch.isfinite(reconstructed).all() and torch.all(error <= tolerance), (error, tolerance)
            return {
                "weights": weights[:4, :4].tolist(),
                "omittedContribution": omitted.tolist(),
                "output": rounded(actual_output[:4].tolist()),
                "inputDimensions": full_input.numel(),
                "omittedDimensions": full_input.numel() - 4,
                "reconstructionTolerance": tolerance.tolist(),
                "reconstructionMaxAbsError": error.max().item(),
            }

        def ops_moe_hook(index):
            def hook(_module, _inputs, output):
                data = ops_cache.pop(index)
                actual_output, logits = output
                torch.testing.assert_close(data["logits"], logits, rtol=0, atol=0)
                selected_logits, selected_experts = logits.topk(config.num_experts_per_tok, dim=-1)
                selected_weights = selected_logits.softmax(dim=-1)
                inverse = data["sortedIndices"].argsort()
                gate, up = data["linearOutput"].chunk(2, dim=-1)
                torch.testing.assert_close(data["product"], data["silu"] * up, rtol=0, atol=0)
                torch.testing.assert_close(data["silu"], torch.nn.functional.silu(gate), rtol=0, atol=0)
                weighted_outputs = data["downOutput"] * data["gates"][:, None]
                reconstructed_full = torch.zeros_like(actual_output.reshape(-1, config.hidden_size)).index_add(
                    0, data["batchIndices"], weighted_outputs)
                torch.testing.assert_close(reconstructed_full, actual_output[0], rtol=0, atol=0)
                token_rows = []
                for token_index in range(logits.shape[0]):
                    expert_rows = []
                    for rank, expert_id in enumerate(selected_experts[token_index].tolist()):
                        row = inverse[token_index * config.num_experts_per_tok + rank].item()
                        torch.testing.assert_close(data["linearInput"][row], data["input"][token_index], rtol=0, atol=0)
                        torch.testing.assert_close(data["gates"][row], selected_weights[token_index, rank], rtol=0, atol=0)
                        gate_weights, up_weights = data["inputWeights"][expert_id].chunk(2, dim=0)
                        expert_rows.append({
                            "id": expert_id,
                            "gate": project(gate_weights, data["linearInput"][row], gate[row]),
                            "up": project(up_weights, data["linearInput"][row], up[row]),
                            "siluGate": rounded(data["silu"][row, :4].tolist()),
                            "product": rounded(data["product"][row, :4].tolist()),
                            "down": project(data["outputWeights"][expert_id], data["product"][row], data["downOutput"][row]),
                            "weightedOutput": rounded(weighted_outputs[row, :4].tolist()),
                        })
                    token_rows.append({
                        "inputPreview": rounded(data["input"][token_index, :4].tolist()),
                        "selectedExperts": selected_experts[token_index].tolist(),
                        "routingWeights": rounded(selected_weights[token_index].tolist()),
                        "experts": expert_rows,
                        "combinedOutput": rounded(actual_output[0, token_index, :4].tolist()),
                    })
                ops_layers[index] = {"index": index, "tokens": token_rows}
            return hook

        attn_cache = {}
        layer_records = {}
        shared = {}
        active_layer = None
        original_rotary = implementation.apply_rotary_pos_emb
        original_repeat = implementation.repeat_kv

        def snapshot_tensor(tensor):
            return tensor.detach().cpu().clone()

        @wraps(original_rotary)
        def observe_rotary(query, key, cos, sin, *positional, **keyword):
            assert active_layer is not None, "Each actual rotary call must belong to an observed attention layer"
            result = original_rotary(query, key, cos, sin, *positional, **keyword)
            attn_cache[active_layer].update(
                queryBefore=snapshot_tensor(query), keyBefore=snapshot_tensor(key),
                queryAfter=snapshot_tensor(result[0]), keyAfter=snapshot_tensor(result[1]),
                cos=snapshot_tensor(cos), sin=snapshot_tensor(sin), repetitions=[],
            )
            return result

        @wraps(original_repeat)
        def observe_repeat(hidden, groups):
            assert active_layer is not None
            result = original_repeat(hidden, groups)
            attn_cache[active_layer]["repetitions"].append((snapshot_tensor(hidden), snapshot_tensor(result), groups))
            return result

        def before_attention(index):
            def hook(_module, _args, kwargs):
                nonlocal active_layer
                assert active_layer is None
                active_layer = index
                attn_cache[index] = {"positions": snapshot_tensor(kwargs["position_ids"])}
            return hook

        def attn_projection_hook(index, kind):
            def hook(_module, _args, output):
                attn_cache[index][f"{kind}Projection"] = snapshot_tensor(output)
            return hook

        def after_attention(index):
            def hook(module, _args, _kwargs, output):
                nonlocal active_layer
                data = attn_cache.pop(index)
                assert active_layer == index
                active_layer = None
                batch, length, _ = data["queryProjection"].shape
                assert batch == 1
                query_heads = data["queryProjection"].view(batch, length, config.num_attention_heads, head_size).transpose(1, 2)
                key_heads = data["keyProjection"].view(batch, length, config.num_key_value_heads, head_size).transpose(1, 2)
                value_heads = data["valueProjection"].view(batch, length, config.num_key_value_heads, head_size).transpose(1, 2)
                torch.testing.assert_close(query_heads, data["queryBefore"], rtol=0, atol=0)
                torch.testing.assert_close(key_heads, data["keyBefore"], rtol=0, atol=0)
                assert len(data["repetitions"]) == 2
                key_input, repeated_key, key_groups = data["repetitions"][0]
                value_input, repeated_value, value_groups = data["repetitions"][1]
                assert key_groups == value_groups == group_size
                torch.testing.assert_close(key_input, data["keyAfter"], rtol=0, atol=0)
                torch.testing.assert_close(value_input, value_heads, rtol=0, atol=0)
                for query_head in range(config.num_attention_heads):
                    unique_head = query_head // group_size
                    torch.testing.assert_close(repeated_key[:, query_head], data["keyAfter"][:, unique_head], rtol=0, atol=0)
                    torch.testing.assert_close(repeated_value[:, query_head], value_heads[:, unique_head], rtol=0, atol=0)
                half = head_size // 2
                torch.testing.assert_close(data["cos"][..., :half], data["cos"][..., half:], rtol=0, atol=0)
                torch.testing.assert_close(data["sin"][..., :half], data["sin"][..., half:], rtol=0, atol=0)
                for name in ("query", "key"):
                    before = data[f"{name}Before"]
                    first = before[..., :half] * data["cos"][:, None, :, :half] - before[..., half:] * data["sin"][:, None, :, :half]
                    second = before[..., half:] * data["cos"][:, None, :, :half] + before[..., :half] * data["sin"][:, None, :, :half]
                    torch.testing.assert_close(torch.cat((first, second), dim=-1), data[f"{name}After"], rtol=0, atol=0)
                scores = data["queryAfter"] @ repeated_key.transpose(2, 3) * module.scaling
                mask = torch.ones((length, length), dtype=torch.bool).triu(1)
                probabilities = scores.masked_fill(mask, torch.finfo(scores.dtype).min).softmax(dim=-1)
                torch.testing.assert_close(probabilities, output[1].detach().cpu(), rtol=0, atol=0)
                shared_values = {"positions": data["positions"][0].tolist(),
                                 "cos": rounded(data["cos"][0, :, :half].tolist()),
                                 "sin": rounded(data["sin"][0, :, :half].tolist())}
                if shared:
                    assert shared == shared_values, "Every layer must use the actual shared position encoding"
                else:
                    shared.update(shared_values)
                layer_records[index] = {
                    "index": index,
                    **{name: rounded(data[name][0].tolist()) for name in
                       ("queryBefore", "keyBefore", "queryAfter", "keyAfter")},
                    "value": rounded(value_heads[0].tolist()),
                }
            return hook


        try:
            for index, layer in enumerate(model.model.layers):
                moe = layer.block_sparse_moe
                handles.extend([
                    layer.self_attn.q_proj.register_forward_hook(projection_hook(index, "query")),
                    layer.self_attn.k_proj.register_forward_hook(projection_hook(index, "key")),
                    layer.self_attn.v_proj.register_forward_hook(projection_hook(index, "value")),
                    layer.self_attn.register_forward_hook(attention_hook(index), with_kwargs=True),
                    moe.router.register_forward_hook(router_hook(index)),
                    moe.output_linear.register_forward_hook(expert_hook(index)),
                    moe.register_forward_hook(moe_hook(index)),
                ])

            for index, layer in enumerate(model.model.layers):
                moe = layer.block_sparse_moe
                handles.extend([
                    moe.router.register_forward_hook(ops_router_hook(index)),
                    moe.input_linear.register_forward_hook(ops_input_hook(index)),
                    moe.activation.register_forward_hook(ops_activation_hook(index)),
                    moe.output_linear.register_forward_hook(ops_output_hook(index)),
                    moe.register_forward_hook(ops_moe_hook(index)),
                ])

            for index, layer in enumerate(model.model.layers):
                attention = layer.self_attn
                handles.extend([
                    attention.register_forward_pre_hook(before_attention(index), with_kwargs=True),
                    attention.q_proj.register_forward_hook(attn_projection_hook(index, "query")),
                    attention.k_proj.register_forward_hook(attn_projection_hook(index, "key")),
                    attention.v_proj.register_forward_hook(attn_projection_hook(index, "value")),
                    attention.register_forward_hook(after_attention(index), with_kwargs=True),
                ])
            implementation.apply_rotary_pos_emb = observe_rotary
            implementation.repeat_kv = observe_repeat

            with torch.inference_mode():
                output = model(**encoded, use_cache=False, output_router_logits=True,
                               output_attentions=True, output_hidden_states=True, return_dict=True)
        finally:
            implementation.apply_rotary_pos_emb = original_rotary
            implementation.repeat_kv = original_repeat
            for handle in handles:
                handle.remove()
        captured_logits = output.logits.detach().cpu().clone()
        forward_seconds = time.monotonic() - start
        with torch.inference_mode():
            plain_output = model(**encoded, use_cache=False, return_dict=True)
        torch.testing.assert_close(captured_logits, plain_output.logits.detach().cpu(), rtol=0, atol=0)
        assert len(current_layers) == len(ops_layers) == len(layer_records) == n_layers
        assert len(output.attentions) == n_layers and len(output.hidden_states) == n_layers + 1
        embedding = output.hidden_states[0].detach().cpu()[0]
        final_hidden = output.hidden_states[-1].detach().cpu()[0]
        with torch.inference_mode():
            expected_embedding = model.model.embed_tokens(encoded["input_ids"]) * config.embedding_multiplier
        torch.testing.assert_close(embedding, expected_embedding[0], rtol=0, atol=0)
        for index, router_logits in enumerate(output.router_logits):
            captured = torch.tensor([token["routerLogits"] for token in current_layers[index]["tokens"]])
            torch.testing.assert_close(captured, router_logits.detach().cpu(), rtol=1e-6, atol=1e-7)
        probabilities = captured_logits[0, -1].float().softmax(dim=-1)
        probabilities64 = captured_logits[0, -1].double().softmax(dim=-1)
        # CausalLM returns logits, not a vocabulary softmax. These display
        # probabilities are derived after inference, like the browser's
        # temperature/sampling view. Float64 avoids a large float32 reduction
        # error across 49,152 vocabulary entries; raw logits stay untouched.
        candidate_probabilities, candidate_ids = probabilities64.topk(10)
        example = {
            "id": trace_id, "title": "Your prompt", "category": "Live", "prompt": prompt,
            "tokens": [{"index": index, "id": token_id, "text": labels[index]}
                       for index, token_id in enumerate(input_ids)],
            "layers": [current_layers[index] for index in range(n_layers)],
            "attention": [rounded(matrix[0].tolist()) for matrix in output.attentions],
            "attentionScores": [attention_score_cache[index]["scores"] for index in range(n_layers)],
            "attentionMaskedScores": [attention_score_cache[index]["masked"] for index in range(n_layers)],
            "embeddingPreview": rounded(embedding[:, :8].tolist()),
            "embeddingNorms": rounded(embedding.norm(dim=-1).tolist()),
            "finalHiddenPreview": rounded(final_hidden[:, :8].tolist()),
            "finalHiddenNorms": rounded(final_hidden.norm(dim=-1).tolist()),
            "nextTokenCandidates": [{"id": token_id, "text": self.vocabulary[token_id],
                                     "probability": round(probability, 9)}
                                    for token_id, probability in zip(candidate_ids.tolist(), candidate_probabilities.tolist())],
            "nextTokenLogits": rounded(captured_logits[0, -1].tolist()),
            "validation": {
                "capturedTokenCount": len(input_ids), "capturedLayerCount": n_layers,
                "forwardSeconds": round(forward_seconds, 3),
                "hookedVsUnhookedMaxLogitError": (captured_logits - plain_output.logits).abs().max().item(),
                "vocabularyProbabilityDtype": "float64",
                "vocabularyFloat64ProbabilitySum": probabilities64.sum().item(),
                "vocabularyFloat32ProbabilitySum": probabilities.double().sum().item(),
                "vocabularyFloat32VsFloat64MaxDifference": (probabilities.double() - probabilities64).abs().max().item(),
                "maxFullVectorReconstructionError": max(token["reconstructionMaxAbsError"]
                                                        for layer in current_layers.values() for token in layer["tokens"]),
            },
        }
        provenance = {
            "timestamp": datetime.now(timezone.utc).isoformat(), "device": "cpu", "dtype": "float32",
            "python": platform.python_version(), "packages": self.packages,
            "script": "scripts/live_capture.py", "scriptSha256": sha256_file(Path(__file__)),
            "sourceScripts": [{"name": name, "sha256": sha256_file(Path(__file__).with_name(name))}
                              for name in ("capture_moe_traces.py", "capture_attention_operations.py", "capture_expert_operations.py")],
        }
        bundle = {
            "schemaVersion": 1, "model": self.model_metadata,
            "capture": {
                **provenance, "mode": "live", "platform": platform.platform(),
                "modelFiles": self.model_files, "configSha256": self.config_sha256,
                "attentionImplementation": "eager", "seed": 0, "threads": self.threads,
                "nextTokenCandidatesProbabilityDtype": "float64",
                "notes": [
                    "Actual execution of the pinned pretrained model in eval mode on the exact supplied prompt; no simulated values.",
                    "One full instrumented pass observes every token, all layers, actual router gates and expert activations. A second uninstrumented pass must match all vocabulary logits at all input positions exactly.",
                    "Scores are reconstructed from actual Q/K projections and actual RoPE/mask inputs. Their softmax must exactly match the returned attention probabilities.",
                    "Transparent wrappers observe original apply_rotary_pos_emb and repeat_kv inputs/outputs and return the original tensors unchanged. V bypass and shared K/V are verified exactly.",
                    "MoE sums use actual routing weights and expert outputs. All 1024 coordinates are checked before rounding. Independent float64 omitted dot products use real weights and full activations.",
                    "Input display labels preserve UTF-8 and whitespace. When a character spans multiple byte-level tokens, incomplete bytes are buffered until the completing token; token IDs and rows are never dropped.",
                    "Vector previews show the first 8 coordinates; expert-operation tiles show 4. Full-vector norms and validation use every coordinate.",
                    "Granite uses RMSNorm, 16 query heads, 8 shared KV heads, head size 64, RoPE, attention scaling 1/64, embedding multiplier 12, residual multiplier 0.22, and logits division by 6.",
                    "All final-token vocabulary logits are captured unchanged for browser-side temperature and sampling calculations. No generated continuation is included in this trace.",
                    "nextTokenCandidates probabilities are derived with a full-vocabulary float64 softmax of the actual captured logits; they are not a float32 tensor returned by the model. This matches the browser precision and avoids accumulated float32 denominator error across 49,152 entries. Separate diagnostics retain the float32 probability mass and its measured difference from float64.",
                    "Numeric arrays are rounded to 9 decimal places. Full-vector validation is performed before rounding and serialized arithmetic is independently validated afterward.",
                ],
            },
            "examples": [example], "vocabulary": self.vocabulary,
        }
        bundle["validation"] = validate_bundle(bundle)
        main_serialized = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        import hashlib
        main_hash = hashlib.sha256(main_serialized.encode("utf-8")).hexdigest()
        sidecar_capture = {**provenance, "mainBundleSha256": main_hash,
                           "mainCaptureScriptSha256": provenance["scriptSha256"],
                           "matchesMainVocabularyLogits": True, "sameForwardPassAsMain": True}
        attention_sidecar = {
            "schemaVersion": 1, "kind": "attention-operations",
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION,
                      "attentionHeads": config.num_attention_heads, "keyValueHeads": config.num_key_value_heads,
                      "headSize": head_size, "ropeTheta": config.rope_theta, "attentionMultiplier": config.attention_multiplier},
            "capture": {**sidecar_capture, "actualValueUnchangedByRotation": True,
                        "actualGroupedKeyValueRepetitionExact": True, "projectionHeadReshapeExact": True,
                        "notes": ["Actual original-function inputs/outputs from the same forward pass as main.json; Q/K reshaping, RoPE, V bypass and shared K/V are checked exactly.",
                                  "cos/sin retain the first half of the actual rotary tensors; both halves are verified identical. All head coordinates are recorded."]},
            "example": {key: example[key] for key in ("id", "prompt", "tokens")},
            "positions": shared["positions"], "rotary": {"cos": shared["cos"], "sin": shared["sin"]},
            "layers": [layer_records[index] for index in range(n_layers)],
        }
        attention_sidecar["validation"] = validate_attention(attention_sidecar, bundle)
        expert_sidecar = {
            "schemaVersion": 1, "kind": "expert-operations",
            "model": {key: self.model_metadata[key] for key in
                      ("id", "revision", "hiddenSize", "intermediateSize", "topK", "numLayers", "numExperts")},
            "capture": {**sidecar_capture, "omittedDotProductDtype": "float64",
                        "notes": ["Actual weights, gate/up outputs, SiLU activations, down-projection inputs and outputs, gates and combined outputs from the same forward pass as main.json.",
                                  "Omitted contributions are independently computed signed float64 dot products over real coordinates 4 onward, never output minus visible terms.",
                                  "Projection tolerance is 1e-5 + 1e-6 times the sum of absolute full dot-product terms. The complete grouped weighted sum matches the actual MoE output exactly."]},
            "example": {key: example[key] for key in ("id", "prompt", "tokens")},
            "layers": [ops_layers[index] for index in range(n_layers)],
        }
        expert_sidecar["validation"] = validate_experts(expert_sidecar, bundle)
        documents = {"main": main_serialized,
                     "attention": json.dumps(attention_sidecar, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n",
                     "expert": json.dumps(expert_sidecar, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"}
        output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        try:
            for kind, content in documents.items():
                path = output_dir / files[kind]
                temporary = path.with_suffix(".json.tmp")
                with temporary.open("x", encoding="utf-8") as handle:
                    written.append(temporary)
                    handle.write(content)
                temporary.replace(path)
                written.append(path)
        except BaseException:
            for path in written:
                path.unlink(missing_ok=True)
            raise
        return {"traceId": trace_id, "prompt": prompt, "tokenCount": len(input_ids),
                "modelId": MODEL_ID, "modelRevision": MODEL_REVISION,
                "seconds": round(time.monotonic() - start, 3), "files": files,
                "validation": {"main": bundle["validation"], "attention": attention_sidecar["validation"],
                               "expert": expert_sidecar["validation"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--trace-id", default="live-cli")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    engine = LiveTraceEngine(args.cache_dir, threads=args.threads)
    print(json.dumps(engine.capture(args.prompt, args.trace_id, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
