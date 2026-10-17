"""Run real-model coverage with MOE_TEST_MODEL_CACHE=/path/to/pinned/cache."""
from __future__ import annotations

import copy
import hashlib
import math
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from live_capture import (MAX_PROMPT_BYTES, MAX_TOKENS, VOCABULARY_PROBABILITY_TOLERANCE,
                          validate_bundle, validate_prompt, validate_trace_id, validate_vocabulary_candidates)


class PromptValidationTests(unittest.TestCase):
    def test_exact_whitespace_and_unicode_are_preserved(self):
        prompt = "  Café 🦉\n\tanswer: "
        self.assertEqual(validate_prompt(prompt), prompt)

    def test_invalid_prompts_are_rejected(self):
        for prompt in (None, "", " \n\t", "\ud800", "x" * (MAX_PROMPT_BYTES + 1)):
            with self.subTest(prompt=repr(prompt)[:50]), self.assertRaises(ValueError):
                validate_prompt(prompt)

    def test_trace_ids_cannot_escape_the_request_directory(self):
        self.assertEqual(validate_trace_id("live-abc123"), "live-abc123")
        for trace_id in ("../trace", "/tmp/trace", "abc/trace", "", ".", "a" * 97):
            with self.subTest(trace_id=trace_id), self.assertRaises(ValueError):
                validate_trace_id(trace_id)


class VocabularyProbabilityTests(unittest.TestCase):
    def setUp(self):
        # exp(log(3)), exp(0), exp(log(1/2)) sum to 9/2, so the exact
        # probabilities are 2/3, 2/9 and 1/9 independently of implementation.
        self.vocabulary = ["a", "b", "c"]
        self.example = {
            "nextTokenLogits": [round(math.log(3), 9), 0.0, round(math.log(0.5), 9)],
            "nextTokenCandidates": [{"id": index, "text": text, "probability": round(probability, 9)}
                                    for index, (text, probability) in enumerate(zip(self.vocabulary, (2/3, 2/9, 1/9)))],
        }

    def test_known_distribution_passes_serialization_bound(self):
        error = validate_vocabulary_candidates(self.example, self.vocabulary)
        self.assertLessEqual(error, VOCABULARY_PROBABILITY_TOLERANCE)

    def test_probability_corruption_smaller_than_old_tolerance_is_rejected(self):
        self.example["nextTokenCandidates"][0]["probability"] -= 1e-5
        with self.assertRaisesRegex(AssertionError, "Vocabulary probability error"):
            validate_vocabulary_candidates(self.example, self.vocabulary)

    def test_changed_logit_is_rejected(self):
        self.example["nextTokenLogits"][0] += 0.01
        with self.assertRaisesRegex(AssertionError, "Vocabulary probability error"):
            validate_vocabulary_candidates(self.example, self.vocabulary)

    def test_duplicate_token_ids_are_rejected(self):
        self.example["nextTokenCandidates"][1]["id"] = 0
        with self.assertRaisesRegex(AssertionError, "IDs must be unique"):
            validate_vocabulary_candidates(self.example, self.vocabulary)

    def test_non_top_candidates_are_rejected(self):
        vocabulary = [str(index) for index in range(11)]
        example = {"nextTokenLogits": list(range(11)),
                   "nextTokenCandidates": [{"id": index, "text": vocabulary[index], "probability": 0.01}
                                           for index in range(10)]}
        with self.assertRaisesRegex(AssertionError, "top vocabulary logits"):
            validate_vocabulary_candidates(example, vocabulary)


@unittest.skipUnless(os.environ.get("MOE_TEST_MODEL_CACHE"), "set MOE_TEST_MODEL_CACHE for real model coverage")
class RealCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from live_capture import LiveTraceEngine
        cls.engine = LiveTraceEngine(os.environ["MOE_TEST_MODEL_CACHE"])
        cls.storage = tempfile.TemporaryDirectory(prefix="moe-live-test-")
        cls.root = Path(cls.storage.name)
        cls.model_identity = id(cls.engine.model)

    @classmethod
    def tearDownClass(cls):
        cls.storage.cleanup()

    def capture(self, prompt, trace_id):
        directory = self.root / trace_id
        result = self.engine.capture(prompt, trace_id, directory)
        self.assertEqual(id(self.engine.model), self.model_identity)
        self.assertTrue(all(result["validation"][kind]["passed"] for kind in ("main", "attention", "expert")))
        main_bytes = (directory / "main.json").read_bytes()
        main = json.loads(main_bytes)
        for kind in ("attention", "expert"):
            sidecar = json.loads((directory / result["files"][kind]).read_text())
            self.assertEqual(sidecar["capture"]["mainBundleSha256"], hashlib.sha256(main_bytes).hexdigest())
            self.assertEqual(sidecar["example"]["prompt"], prompt)
            self.assertEqual(sidecar["example"]["tokens"], main["examples"][0]["tokens"])
        return result, main

    def test_baseline_matches_the_independent_recorded_trace(self):
        _, main = self.capture("The capital of France is", "live-baseline")
        reference = json.loads((ROOT / "static/traces/granite-moe.json").read_text())["examples"][0]
        example = main["examples"][0]
        for key in ("prompt", "tokens", "nextTokenLogits", "attention", "attentionScores", "layers"):
            self.assertEqual(example[key], reference[key], key)
        self.assertEqual(example["validation"]["hookedVsUnhookedMaxLogitError"], 0)

    def test_capital_of_france_probabilities_match_real_logits_in_float64(self):
        import torch
        result, main = self.capture("The capital of France", "live-probability-regression")
        example = main["examples"][0]
        self.assertEqual(example["prompt"], "The capital of France")
        self.assertEqual(example["validation"]["hookedVsUnhookedMaxLogitError"], 0)
        self.assertEqual(example["validation"]["vocabularyProbabilityDtype"], "float64")
        self.assertEqual(main["capture"]["nextTokenCandidatesProbabilityDtype"], "float64")
        self.assertLessEqual(result["validation"]["main"]["maxVocabularySoftmaxFloat64ReconstructionError"],
                             VOCABULARY_PROBABILITY_TOLERANCE)
        # Independently derive the probabilities from the complete persisted
        # logits, using Python's accurately summed normalization denominator.
        logits = example["nextTokenLogits"]
        maximum = max(logits)
        denominator = math.fsum(math.exp(value - maximum) for value in logits)
        for candidate in example["nextTokenCandidates"]:
            expected = math.exp(logits[candidate["id"]] - maximum) / denominator
            self.assertAlmostEqual(candidate["probability"], expected, delta=VOCABULARY_PROBABILITY_TOLERANCE)
        # The former float32 display calculation remains an honest diagnostic,
        # rather than being silently relabeled as high precision.
        saved = torch.tensor(logits, dtype=torch.float32)
        probability32, probability64 = saved.softmax(-1), saved.double().softmax(-1)
        self.assertAlmostEqual(example["validation"]["vocabularyFloat32ProbabilitySum"],
                               probability32.double().sum().item(), delta=1e-12)
        self.assertAlmostEqual(example["validation"]["vocabularyFloat32VsFloat64MaxDifference"],
                               (probability32.double() - probability64).abs().max().item(), delta=1e-12)
        for field in ("probability", "logit"):
            tampered = copy.deepcopy(main)
            first = tampered["examples"][0]["nextTokenCandidates"][0]
            if field == "probability":
                first["probability"] -= 1e-5
            else:
                tampered["examples"][0]["nextTokenLogits"][first["id"]] += 0.01
            with self.subTest(tampered=field), self.assertRaisesRegex(AssertionError, "Vocabulary probability error"):
                validate_bundle(tampered)

    def test_unicode_and_whitespace_use_real_ids_without_replacement_characters(self):
        prompt = "  Café 🦉\n\tanswer: "
        result, main = self.capture(prompt, "live-unicode")
        example = main["examples"][0]
        self.assertEqual(example["prompt"], prompt)
        self.assertEqual("".join(token["text"] for token in example["tokens"]), prompt)
        self.assertEqual([token["id"] for token in example["tokens"]],
                         self.engine.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        self.assertEqual(result["tokenCount"], len(example["tokens"]))
        self.assertEqual(example["validation"]["hookedVsUnhookedMaxLogitError"], 0)

    def test_limit_rejects_before_model_execution_or_file_creation(self):
        prompt = "word " * (MAX_TOKENS + 1)
        directory = self.root / "live-too-long"
        with self.assertRaisesRegex(ValueError, "at most 32 tokens"):
            self.engine.capture(prompt, "live-too-long", directory)
        self.assertFalse(directory.exists())

    def test_failed_forward_restores_observers_and_removes_hooks(self):
        from transformers.models.granitemoe import modeling_granitemoe as implementation
        original_rotary = implementation.apply_rotary_pos_emb
        original_repeat = implementation.repeat_kv
        hooks_before = sum(len(module._forward_hooks) + len(module._forward_pre_hooks)
                           for module in self.engine.model.modules())
        def fail(_module, _inputs):
            raise RuntimeError("injected capture interruption")
        handle = self.engine.model.model.layers[0].register_forward_pre_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected capture interruption"):
                self.engine.capture("hello", "live-interrupted", self.root / "live-interrupted")
        finally:
            handle.remove()
        self.assertIs(implementation.apply_rotary_pos_emb, original_rotary)
        self.assertIs(implementation.repeat_kv, original_repeat)
        self.assertEqual(sum(len(module._forward_hooks) + len(module._forward_pre_hooks)
                             for module in self.engine.model.modules()), hooks_before)
        self.assertFalse((self.root / "live-interrupted").exists())


if __name__ == "__main__":
    unittest.main()
