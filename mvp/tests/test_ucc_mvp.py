from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from ucc_mvp.engine import (
    MVP_NOT_APPLICABLE,
    ObservationFailed,
    ObservationIndeterminate,
    TransitionRetryableError,
    UccMvpEngine,
)


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "formal" / "ucc-2.0.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)


class RetryableTransitionEngine(UccMvpEngine):
    def _execute_transition(self, target, desired_state):
        raise TransitionRetryableError("simulated transient transition failure")


class ObservationFailedEngine(UccMvpEngine):
    def _observe_target(self, target):
        raise ObservationFailed("simulated observation failure")


class ObservationIndeterminateEngine(UccMvpEngine):
    def _observe_target(self, target):
        raise ObservationIndeterminate("simulated observation indeterminate")


class UccMvpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def execute(self, message, engine=None):
        engine = engine or UccMvpEngine(now_fn=lambda: "2026-03-22T12:00:00Z")
        result = engine.execute(message)
        errors = sorted(VALIDATOR.iter_errors(result), key=lambda err: err.json_path)
        self.assertEqual([], errors, msg="\n".join(error.message for error in errors))
        return result

    def message(self, target, desired_state, mode="apply", requires=None):
        declaration = {
            "target": str(target),
            "desired_state": desired_state,
        }
        if mode != "apply":
            declaration["mode"] = mode
        if requires is not None:
            declaration["requires"] = requires

        return {
            "meta": {
                "contract": "ucc-mvp",
                "version": "2.0",
                "id": f"case-{Path(target).name}",
                "timestamp": "2026-03-22T11:59:00Z",
            },
            "declaration": declaration,
        }

    def test_declaration_message_validates_against_schema(self):
        message = self.message(self.root / "hello.txt", {"exists": True, "content": "hello\n"})
        errors = sorted(VALIDATOR.iter_errors(message), key=lambda err: err.json_path)
        self.assertEqual([], errors, msg="\n".join(error.message for error in errors))

    def test_phase_1_happy_clean(self):
        target = self.root / "hello.txt"
        result = self.execute(self.message(target, {"exists": True, "content": "hello\n"}))

        self.assertEqual("ok", result["result"]["observation"])
        self.assertEqual("changed", result["result"]["outcome"])
        self.assertEqual("complete", result["result"]["completion"])
        self.assertEqual({"exists": False}, result["observe"]["observed_before"])
        self.assertEqual("hello\n", target.read_text(encoding="utf-8"))
        self.assertEqual("hello\n", result["observe"]["observed_after"]["content"])
        self.assertIn("content", result["observe"]["diff"])

    def test_phase_2_happy_idempotent(self):
        target = self.root / "hello.txt"
        target.write_text("hello\n", encoding="utf-8")

        result = self.execute(self.message(target, {"exists": True, "content": "hello\n"}))

        self.assertEqual("ok", result["result"]["observation"])
        self.assertEqual("converged", result["result"]["outcome"])
        self.assertEqual({}, result["observe"]["diff"])
        self.assertNotIn("observed_after", result["observe"])

    def test_phase_3_fail_retryable(self):
        target = self.root / "hello.txt"
        result = self.execute(
            self.message(target, {"exists": True, "content": "hello\n"}),
            engine=RetryableTransitionEngine(now_fn=lambda: "2026-03-22T12:00:00Z"),
        )

        self.assertEqual("ok", result["result"]["observation"])
        self.assertEqual("failed", result["result"]["outcome"])
        self.assertEqual("retryable", result["result"]["failure_class"])
        self.assertIn("simulated transient", result["result"]["message"])
        self.assertIn("diff", result["observe"])
        self.assertNotIn("observed_after", result["observe"])

    def test_phase_4_fail_permanent(self):
        target = self.root / "hello.txt"
        result = self.execute(
            self.message(target, {"exists": False, "content": "invalid"}),
        )

        self.assertEqual("ok", result["result"]["observation"])
        self.assertEqual("failed", result["result"]["outcome"])
        self.assertEqual("permanent", result["result"]["failure_class"])
        self.assertIn("forbidden", result["result"]["message"])
        self.assertIn("observed_before", result["observe"])
        self.assertNotIn("diff", result["observe"])

    def test_phase_5_unchanged(self):
        target = self.root / "hello.txt"
        result = self.execute(
            self.message(target, {"exists": True, "content": "hello\n"}, mode="dry_run"),
        )

        self.assertEqual("ok", result["result"]["observation"])
        self.assertEqual("unchanged", result["result"]["outcome"])
        self.assertEqual("dry_run", result["result"]["inhibitor"])
        self.assertNotIn("observed_after", result["observe"])
        self.assertFalse(target.exists())

    def test_phase_6_partial_not_applicable(self):
        self.assertIn("Phase 6", MVP_NOT_APPLICABLE)
        self.assertIn("Single-target", MVP_NOT_APPLICABLE["Phase 6"])

    def test_phase_7_fail_observation(self):
        target = self.root / "hello.txt"
        result = self.execute(
            self.message(target, {"exists": True, "content": "hello\n"}),
            engine=ObservationFailedEngine(now_fn=lambda: "2026-03-22T12:00:00Z"),
        )

        self.assertEqual("failed", result["result"]["observation"])
        self.assertNotIn("outcome", result["result"])
        self.assertEqual({}, result["observe"])

    def test_phase_8_observation_indeterminate(self):
        target = self.root / "hello.txt"
        result = self.execute(
            self.message(target, {"exists": True, "content": "hello\n"}),
            engine=ObservationIndeterminateEngine(now_fn=lambda: "2026-03-22T12:00:00Z"),
        )

        self.assertEqual("indeterminate", result["result"]["observation"])
        self.assertNotIn("outcome", result["result"])
        self.assertEqual({}, result["observe"])

    def test_pre_1_unsatisfied_precondition(self):
        target = self.root / "hello.txt"
        dependency = self.root / "dependency.txt"
        result = self.execute(
            self.message(
                target,
                {"exists": True, "content": "hello\n"},
                requires=[
                    {
                        "declaration_id": "ensure-dependency",
                        "target": str(dependency),
                        "desired_state": {"exists": True},
                    }
                ],
            )
        )

        self.assertEqual("ok", result["result"]["observation"])
        self.assertEqual("failed", result["result"]["outcome"])
        self.assertEqual("permanent", result["result"]["failure_class"])
        self.assertFalse(result["observe"]["preconditions"]["satisfied"])
        self.assertIn("ensure-dependency", result["result"]["message"])
        self.assertNotIn("observed_before", result["observe"])
        self.assertNotIn("diff", result["observe"])

    def test_pre_2_cycle_not_applicable(self):
        self.assertIn("PRE-2", MVP_NOT_APPLICABLE)
        self.assertIn("Single-message", MVP_NOT_APPLICABLE["PRE-2"])

    def test_dry_1_suppression(self):
        target = self.root / "hello.txt"
        result = self.execute(
            self.message(target, {"exists": True, "content": "hello\n"}, mode="dry_run"),
        )

        self.assertEqual("unchanged", result["result"]["outcome"])
        self.assertEqual("dry_run", result["result"]["inhibitor"])
        self.assertGreaterEqual(len(result["observe"]["diff"]), 1)
        self.assertFalse(target.exists())

    def test_dry_2_converged(self):
        target = self.root / "hello.txt"
        target.write_text("hello\n", encoding="utf-8")
        result = self.execute(
            self.message(target, {"exists": True, "content": "hello\n"}, mode="dry_run"),
        )

        self.assertEqual("converged", result["result"]["outcome"])
        self.assertEqual({}, result["observe"]["diff"])
        self.assertEqual("hello\n", target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
