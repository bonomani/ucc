from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MVP_NOT_APPLICABLE = {
    "Phase 6": "Single-target reference engine; no multi-target partial-convergence path exists.",
    "PRE-2": "Single-message reference engine; dependency-graph cycle detection belongs to orchestrators.",
}


class UccMvpError(Exception):
    pass


class ObservationFailed(UccMvpError):
    pass


class ObservationIndeterminate(UccMvpError):
    pass


class DeclarationPermanentError(UccMvpError):
    pass


class DeclarationRetryableError(UccMvpError):
    pass


class TransitionPermanentError(UccMvpError):
    pass


class TransitionRetryableError(UccMvpError):
    pass


class PreconditionObservationError(UccMvpError):
    def __init__(self, observation: str, report: dict[str, Any], message: str) -> None:
        super().__init__(message)
        self.observation = observation
        self.report = report
        self.message = message


class UccMvpEngine:
    def __init__(self, now_fn=None) -> None:
        self._now_fn = now_fn or self._default_now

    def execute(self, message: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()

        meta_in = message["meta"]
        declaration = message["declaration"]
        target = declaration["target"]
        mode = declaration.get("mode", "apply")

        observe_block: dict[str, Any] = {}

        try:
            preconditions_report = self._evaluate_preconditions(declaration.get("requires", []))
        except PreconditionObservationError as exc:
            observe_block["preconditions"] = exc.report
            return self._build_result(
                meta_in=meta_in,
                observe=observe_block,
                result={
                    "observation": exc.observation,
                    "message": exc.message,
                },
                start=start,
            )

        if preconditions_report is not None:
            observe_block["preconditions"] = preconditions_report
            if not preconditions_report["satisfied"]:
                first_failed = next(
                    check for check in preconditions_report["checks"] if not check["satisfied"]
                )
                return self._build_result(
                    meta_in=meta_in,
                    observe=observe_block,
                    result={
                        "observation": "ok",
                        "outcome": "failed",
                        "failure_class": "permanent",
                        "message": first_failed["message"],
                    },
                    start=start,
                )

        try:
            observed_before = self._observe_target(target)
        except ObservationFailed as exc:
            return self._build_result(
                meta_in=meta_in,
                observe=observe_block,
                result={"observation": "failed", "message": str(exc)},
                start=start,
            )
        except ObservationIndeterminate as exc:
            return self._build_result(
                meta_in=meta_in,
                observe=observe_block,
                result={"observation": "indeterminate", "message": str(exc)},
                start=start,
            )

        observe_block["observed_before"] = observed_before

        try:
            desired_state = self._evaluate_declaration(declaration, observed_before)
        except DeclarationPermanentError as exc:
            return self._build_result(
                meta_in=meta_in,
                observe=observe_block,
                result={
                    "observation": "ok",
                    "outcome": "failed",
                    "failure_class": "permanent",
                    "message": str(exc),
                },
                start=start,
            )
        except DeclarationRetryableError as exc:
            return self._build_result(
                meta_in=meta_in,
                observe=observe_block,
                result={
                    "observation": "ok",
                    "outcome": "failed",
                    "failure_class": "retryable",
                    "message": str(exc),
                },
                start=start,
            )

        diff = self._diff_states(observed_before, desired_state)
        observe_block["diff"] = diff

        if not diff:
            return self._build_result(
                meta_in=meta_in,
                observe=observe_block,
                result={
                    "observation": "ok",
                    "outcome": "converged",
                },
                start=start,
            )

        if mode == "dry_run":
            return self._build_result(
                meta_in=meta_in,
                observe=observe_block,
                result={
                    "observation": "ok",
                    "outcome": "unchanged",
                    "inhibitor": "dry_run",
                    "message": "dry_run suppressed the transition attempt.",
                },
                start=start,
            )

        try:
            proof = self._execute_transition(target, desired_state)
        except TransitionPermanentError as exc:
            return self._build_result(
                meta_in=meta_in,
                observe=observe_block,
                result={
                    "observation": "ok",
                    "outcome": "failed",
                    "failure_class": "permanent",
                    "message": str(exc),
                },
                start=start,
            )
        except TransitionRetryableError as exc:
            return self._build_result(
                meta_in=meta_in,
                observe=observe_block,
                result={
                    "observation": "ok",
                    "outcome": "failed",
                    "failure_class": "retryable",
                    "message": str(exc),
                },
                start=start,
            )

        try:
            observed_after = self._observe_target(target)
        except ObservationFailed as exc:
            return self._build_result(
                meta_in=meta_in,
                observe=observe_block,
                result={
                    "observation": "ok",
                    "outcome": "failed",
                    "failure_class": "retryable",
                    "message": f"Post-mutation observation failed: {exc}",
                },
                start=start,
            )
        except ObservationIndeterminate as exc:
            return self._build_result(
                meta_in=meta_in,
                observe=observe_block,
                result={
                    "observation": "ok",
                    "outcome": "failed",
                    "failure_class": "retryable",
                    "message": f"Post-mutation observation indeterminate: {exc}",
                },
                start=start,
            )

        observed_after_diff = self._diff_states(observed_after, desired_state)
        if observed_after_diff:
            return self._build_result(
                meta_in=meta_in,
                observe=observe_block,
                result={
                    "observation": "ok",
                    "outcome": "failed",
                    "failure_class": "retryable",
                    "message": "Transition completed but post-mutation observation did not verify convergence.",
                },
                start=start,
            )

        observe_block["observed_after"] = observed_after
        return self._build_result(
            meta_in=meta_in,
            observe=observe_block,
            result={
                "observation": "ok",
                "outcome": "changed",
                "completion": "complete",
                "proof": proof,
            },
            start=start,
        )

    def _evaluate_preconditions(self, requires: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not requires:
            return None

        checks = []
        all_satisfied = True

        for requirement in requires:
            expected = self._normalize_desired_state(requirement["desired_state"], allow_minimal=True)
            try:
                observed = self._observe_target(requirement["target"])
            except ObservationFailed as exc:
                check = {
                    "declaration_id": requirement["declaration_id"],
                    "target": requirement["target"],
                    "expected_state": expected,
                    "observation": "failed",
                    "satisfied": False,
                    "message": str(exc),
                }
                report = {"evaluated": True, "satisfied": False, "checks": checks + [check]}
                raise PreconditionObservationError(
                    "failed",
                    report,
                    f"Precondition observation failed for declaration '{requirement['declaration_id']}'.",
                ) from exc
            except ObservationIndeterminate as exc:
                check = {
                    "declaration_id": requirement["declaration_id"],
                    "target": requirement["target"],
                    "expected_state": expected,
                    "observation": "indeterminate",
                    "satisfied": False,
                    "message": str(exc),
                }
                report = {"evaluated": True, "satisfied": False, "checks": checks + [check]}
                raise PreconditionObservationError(
                    "indeterminate",
                    report,
                    f"Precondition observation indeterminate for declaration '{requirement['declaration_id']}'.",
                ) from exc

            satisfied = self._state_satisfies(observed, expected)
            if not satisfied:
                all_satisfied = False

            check = {
                "declaration_id": requirement["declaration_id"],
                "target": requirement["target"],
                "expected_state": expected,
                "observed_state": observed,
                "observation": "ok",
                "satisfied": satisfied,
            }
            if not satisfied:
                check["message"] = (
                    f"Precondition not satisfied for declaration '{requirement['declaration_id']}' "
                    f"on target '{requirement['target']}'."
                )
            checks.append(check)

        return {
            "evaluated": True,
            "satisfied": all_satisfied,
            "checks": checks,
        }

    def _observe_target(self, target: str) -> dict[str, Any]:
        path = Path(target)
        try:
            if path.is_file():
                return {
                    "exists": True,
                    "content": path.read_text(encoding="utf-8"),
                }
            if path.exists():
                return {
                    "exists": True,
                    "content": None,
                    "kind": "non_file",
                }
            return {"exists": False}
        except FileNotFoundError:
            return {"exists": False}
        except PermissionError as exc:
            raise ObservationFailed(f"Could not observe target '{target}': {exc}") from exc
        except OSError as exc:
            raise ObservationIndeterminate(f"Observation of target '{target}' was indeterminate: {exc}") from exc

    def _evaluate_declaration(
        self,
        declaration: dict[str, Any],
        observed_before: dict[str, Any],
    ) -> dict[str, Any]:
        mode = declaration.get("mode", "apply")
        if mode == "verify":
            raise DeclarationPermanentError("mode=verify is outside the MVP scope.")

        desired_state = self._normalize_desired_state(declaration["desired_state"], allow_minimal=False)
        if observed_before.get("kind") == "non_file":
            raise DeclarationPermanentError("Target exists but is not a regular file.")
        return desired_state

    def _normalize_desired_state(self, desired_state: Any, allow_minimal: bool) -> dict[str, Any]:
        if not isinstance(desired_state, dict):
            raise DeclarationPermanentError("desired_state must be an object.")

        exists = desired_state.get("exists")
        if exists is None and "content" in desired_state:
            exists = True

        if not isinstance(exists, bool):
            raise DeclarationPermanentError("desired_state.exists must be a boolean.")

        if exists is False and "content" in desired_state:
            raise DeclarationPermanentError("desired_state.content is forbidden when desired_state.exists=false.")

        normalized: dict[str, Any] = {"exists": exists}

        if "content" in desired_state:
            content = desired_state["content"]
            if not isinstance(content, str):
                raise DeclarationPermanentError("desired_state.content must be a string.")
            normalized["content"] = content

        if exists and not allow_minimal and "content" not in normalized:
            raise DeclarationPermanentError("MVP file declarations require desired_state.content when exists=true.")

        if not normalized:
            raise DeclarationPermanentError("desired_state is empty.")

        return normalized

    def _diff_states(self, observed: dict[str, Any], desired: dict[str, Any]) -> dict[str, Any]:
        delta = {}
        for key in sorted(set(observed) | set(desired)):
            before = observed.get(key)
            after = desired.get(key)
            if before != after:
                delta[key] = {"before": before, "after": after}
        return delta

    def _state_satisfies(self, observed: dict[str, Any], expected: dict[str, Any]) -> bool:
        for key, value in expected.items():
            if observed.get(key) != value:
                return False
        return True

    def _execute_transition(self, target: str, desired_state: dict[str, Any]) -> dict[str, Any]:
        path = Path(target)

        try:
            if not desired_state["exists"]:
                if path.exists():
                    if path.is_dir():
                        raise TransitionPermanentError(f"Target '{target}' is a directory; MVP only manages files.")
                    path.unlink()
                return {"transition": "file_delete", "target": target}

            content = desired_state["content"]
            path.parent.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(path.parent),
                delete=False,
            ) as handle:
                handle.write(content)
                temp_name = handle.name

            os.replace(temp_name, path)
            return {
                "transition": "file_write",
                "target": target,
                "bytes_written": len(content.encode("utf-8")),
            }
        except PermissionError as exc:
            raise TransitionPermanentError(f"Permission denied while mutating '{target}': {exc}") from exc
        except TransitionPermanentError:
            raise
        except OSError as exc:
            raise TransitionRetryableError(f"Transient filesystem failure while mutating '{target}': {exc}") from exc

    def _build_result(
        self,
        meta_in: dict[str, Any],
        observe: dict[str, Any],
        result: dict[str, Any],
        start: float,
    ) -> dict[str, Any]:
        meta_out = {
            "contract": meta_in["contract"],
            "version": "2.0",
            "id": meta_in["id"],
            "timestamp": self._now_fn(),
            "duration_ms": int((time.monotonic() - start) * 1000),
        }

        for key in ("scope", "caused_by"):
            if key in meta_in:
                meta_out[key] = meta_in[key]

        return {
            "meta": meta_out,
            "observe": observe,
            "result": result,
        }

    @staticmethod
    def _default_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
