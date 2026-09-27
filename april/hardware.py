"""Run-local hardware provenance for APRIL experiments.

This module deliberately does not change clocks or power limits.  It records a
``not_requested`` baseline, verifies the observed GPU state with ``nvidia-smi``,
and samples the same device for the lifetime of the context manager.  Command
execution is injectable so the recorder can be tested without a GPU.
"""

from __future__ import annotations

def _public_subprocess_env(source=None):
    """Pass required execution settings without inheriting credential variables."""
    import os
    source = os.environ if source is None else source
    allowed = {
        'PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT', 'TMP', 'TEMP',
        'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
        'CUDA_DEVICE_ORDER', 'CUBLAS_WORKSPACE_CONFIG', 'PYTHONPATH',
        'PYTHONHASHSEED', 'PYTHONIOENCODING', 'PYTHONNOUSERSITE',
        'PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
        'HF_HOME', 'HF_DATASETS_CACHE', 'TRANSFORMERS_CACHE',
        'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TOKENIZERS_PARALLELISM',
    }
    return {key: str(value) for key, value in source.items() if key in allowed}


import csv
import json
import math
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence


REQUIRED_FIELDS = (
    "requested_profile",
    "profile_status",
    "actual_power_limit_w",
    "energy_measurement_source",
    "cap_saturated_ratio",
)

SNAPSHOT_QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "power.limit",
    "power.draw",
    "clocks.current.memory",
    "clocks.current.sm",
    "clocks.current.graphics",
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
    "memory.free",
    "pstate",
)

SAMPLE_QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "power.draw",
    "power.limit",
    "clocks.current.sm",
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
    "pstate",
)

SAMPLE_COLUMNS = (
    "timestamp_unix_s",
    "timestamp_utc",
    "elapsed_s",
    "sample_status",
    "requested_profile",
    "profile_status",
    "actual_power_limit_w",
    "index",
    "uuid",
    "name",
    "power_draw_w",
    "power_limit_w",
    "clocks_sm_mhz",
    "temperature_c",
    "utilization_gpu_pct",
    "memory_used_mib",
    "pstate",
    "query_command_json",
    "query_returncode",
    "query_stdout",
    "query_stderr",
    "error",
)


CommandRunner = Callable[[Sequence[str]], Any]


@dataclass(frozen=True)
class CommandOutcome:
    """Serializable outcome of one exact command invocation."""

    command: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    exception: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.exception is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "command_display": subprocess.list2cmdline(list(self.command)),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exception": self.exception,
            "ok": self.ok,
        }


def _default_command_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
     env=_public_subprocess_env(), timeout=60)


def _utc_iso(timestamp_s: float) -> str:
    return datetime.fromtimestamp(timestamp_s, tz=timezone.utc).isoformat()


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _normalize_outcome(command: Sequence[str], result: Any) -> CommandOutcome:
    if isinstance(result, Mapping):
        returncode = result.get("returncode")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
    else:
        returncode = getattr(result, "returncode", None)
        stdout = getattr(result, "stdout", "")
        stderr = getattr(result, "stderr", "")
    try:
        normalized_returncode = int(returncode) if returncode is not None else None
    except (TypeError, ValueError):
        normalized_returncode = None
    return CommandOutcome(
        command=tuple(str(part) for part in command),
        returncode=normalized_returncode,
        stdout="" if stdout is None else str(stdout),
        stderr="" if stderr is None else str(stderr),
    )


class HardwareProfileRecorder:
    """Record measured GPU state for one experiment run.

    Parameters
    ----------
    run_dir:
        Experiment run directory.  Artifacts are written under ``hardware/``.
    gpu_id:
        The index or UUID passed verbatim to ``nvidia-smi -i``.
    requested_profile:
        This integration is intentionally observation-only and therefore accepts
        only ``"not_requested"``.
    command_runner:
        Optional callable receiving the argv sequence.  Its return value may be a
        ``subprocess.CompletedProcess`` or a mapping with returncode/stdout/stderr.
    """

    def __init__(
        self,
        run_dir: str | Path,
        gpu_id: str | int,
        *,
        requested_profile: str = "not_requested",
        sample_interval_s: float = 1.0,
        command_runner: CommandRunner | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if requested_profile != "not_requested":
            raise ValueError(
                "APRIL HardwareProfileRecorder is observation-only; "
                "requested_profile must be 'not_requested'"
            )
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be positive")

        self.run_dir = Path(run_dir)
        self.hardware_dir = self.run_dir / "hardware"
        self.gpu_id = str(gpu_id)
        self.requested_profile = requested_profile
        self.profile_status = "not_requested"
        self.sample_interval_s = float(sample_interval_s)
        self._command_runner = command_runner or _default_command_runner
        self._clock = clock

        self.snapshot_before_path = self.hardware_dir / "hardware_snapshot_before.json"
        self.snapshot_after_path = self.hardware_dir / "hardware_snapshot_after_apply.json"
        self.samples_path = self.hardware_dir / "hardware_samples.csv"
        self.summary_path = self.hardware_dir / "hardware_summary.json"

        self.actual_power_limit_w: float | None = None
        self.energy_measurement_source: str | None = None
        self.cap_saturated_ratio: float | None = None
        self.summary: dict[str, Any] | None = None

        self._started = False
        self._finalized = False
        self._start_timestamp_s: float | None = None
        self._workload_start_timestamp_s: float | None = None
        self._stop_event = threading.Event()
        self._sample_lock = threading.Lock()
        self._sample_thread: threading.Thread | None = None
        self._samples_file: Any = None
        self._samples_writer: csv.DictWriter[str] | None = None
        self._sample_rows: list[dict[str, Any]] = []
        self._before_snapshot: dict[str, Any] | None = None
        self._after_snapshot: dict[str, Any] | None = None

    def __enter__(self) -> "HardwareProfileRecorder":
        return self.start()

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        self.finalize(workload_exception=exc)
        return False

    def _run(self, command: Sequence[str]) -> CommandOutcome:
        try:
            return _normalize_outcome(command, self._command_runner(tuple(command)))
        except BaseException as exc:  # command failures are measured data
            return CommandOutcome(
                command=tuple(str(part) for part in command),
                returncode=None,
                stdout="",
                stderr="",
                exception=f"{type(exc).__name__}: {exc}",
            )

    def _query(self, fields: Sequence[str]) -> tuple[dict[str, str], CommandOutcome, str | None]:
        command = (
            "nvidia-smi",
            "-i",
            self.gpu_id,
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        )
        outcome = self._run(command)
        if not outcome.ok:
            reason = outcome.exception or outcome.stderr.strip() or "nonzero return code"
            return {}, outcome, f"nvidia-smi query failed: {reason}"
        lines = [line for line in outcome.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            return {}, outcome, f"expected exactly one GPU row, observed {len(lines)}"
        values = next(csv.reader([lines[0]], skipinitialspace=True))
        if len(values) != len(fields):
            return (
                {},
                outcome,
                f"expected {len(fields)} query columns, observed {len(values)}",
            )
        return (
            {field: value.strip() for field, value in zip(fields, values)},
            outcome,
            None,
        )

    def _query_energy_counter(self) -> dict[str, Any]:
        fields = ("total_energy_consumption",)
        values, outcome, error = self._query(fields)
        energy_mj = _to_float(values.get(fields[0])) if not error else None
        if energy_mj is None and error is None:
            error = "total_energy_consumption was not numeric"
        return {
            "supported": energy_mj is not None,
            "total_energy_mj": energy_mj,
            "missing_reason": error,
            "query": outcome.as_dict(),
        }

    def _snapshot(self, stage: str) -> dict[str, Any]:
        timestamp_s = self._clock()
        values, outcome, query_error = self._query(SNAPSHOT_QUERY_FIELDS)
        actual_limit = _to_float(values.get("power.limit"))
        limit_reason = None
        if actual_limit is None:
            limit_reason = query_error or "power.limit was not numeric"
        energy_counter = self._query_energy_counter()
        return {
            "schema_version": "april.hardware_snapshot.v1",
            "stage": stage,
            "timestamp_unix_s": timestamp_s,
            "timestamp_utc": _utc_iso(timestamp_s),
            "gpu_selector": self.gpu_id,
            "requested_profile": self.requested_profile,
            "profile_status": self.profile_status,
            "profile_apply": {
                "attempted": False,
                "command": None,
                "returncode": None,
                "status": "not_requested",
                "reason": "no hardware profile or power/clock change was requested",
            },
            "gpu": values,
            "gpu_identity": {
                "index": values.get("index"),
                "uuid": values.get("uuid"),
                "name": values.get("name"),
            },
            "actual_power_limit_w": actual_limit,
            "actual_power_limit_missing_reason": limit_reason,
            "verification_query": outcome.as_dict(),
            "verification_error": query_error,
            "nvml_energy_counter_supported": energy_counter["supported"],
            "energy_counter": energy_counter,
        }

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def start(self) -> "HardwareProfileRecorder":
        if self._started:
            raise RuntimeError("HardwareProfileRecorder.start() may only be called once")
        self._started = True
        self.hardware_dir.mkdir(parents=True, exist_ok=True)

        # The first artifact is emitted before even the no-op profile decision.
        self._before_snapshot = self._snapshot("before_profile_apply")
        self._write_json(self.snapshot_before_path, self._before_snapshot)

        # There is no apply command for the observation-only baseline, but a new
        # verification query is still required immediately after that decision.
        self._after_snapshot = self._snapshot("after_profile_apply_verification")
        self.actual_power_limit_w = self._after_snapshot["actual_power_limit_w"]
        self._write_json(self.snapshot_after_path, self._after_snapshot)

        self._samples_file = self.samples_path.open("w", newline="", encoding="utf-8")
        self._samples_writer = csv.DictWriter(
            self._samples_file,
            fieldnames=list(SAMPLE_COLUMNS),
            extrasaction="ignore",
        )
        self._samples_writer.writeheader()
        self._samples_file.flush()

        self._start_timestamp_s = self._clock()
        self._collect_sample()
        self._sample_thread = threading.Thread(
            target=self._sample_loop,
            name=f"april-hardware-sampler-{self.gpu_id}",
            daemon=True,
        )
        self._sample_thread.start()
        # ``__enter__`` returns immediately after this timestamp, so the first
        # synchronous sample precedes the measured workload.
        self._workload_start_timestamp_s = self._clock()
        return self

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_s):
            self._collect_sample()

    def _collect_sample(self) -> None:
        with self._sample_lock:
            timestamp_s = self._clock()
            values, outcome, error = self._query(SAMPLE_QUERY_FIELDS)
            row: dict[str, Any] = {
                "timestamp_unix_s": timestamp_s,
                "timestamp_utc": _utc_iso(timestamp_s),
                "elapsed_s": (
                    max(0.0, timestamp_s - self._start_timestamp_s)
                    if self._start_timestamp_s is not None
                    else 0.0
                ),
                "sample_status": "ok" if error is None else "error",
                "requested_profile": self.requested_profile,
                "profile_status": self.profile_status,
                "actual_power_limit_w": self.actual_power_limit_w,
                "index": values.get("index"),
                "uuid": values.get("uuid"),
                "name": values.get("name"),
                "power_draw_w": _to_float(values.get("power.draw")),
                "power_limit_w": _to_float(values.get("power.limit")),
                "clocks_sm_mhz": _to_float(values.get("clocks.current.sm")),
                "temperature_c": _to_float(values.get("temperature.gpu")),
                "utilization_gpu_pct": _to_float(values.get("utilization.gpu")),
                "memory_used_mib": _to_float(values.get("memory.used")),
                "pstate": values.get("pstate"),
                "query_command_json": json.dumps(list(outcome.command), ensure_ascii=False),
                "query_returncode": outcome.returncode,
                "query_stdout": outcome.stdout,
                "query_stderr": outcome.stderr,
                "error": error or outcome.exception,
            }
            self._sample_rows.append(row)
            if self._samples_writer is not None and self._samples_file is not None:
                self._samples_writer.writerow(row)
                self._samples_file.flush()

    def _sample_energy(self) -> tuple[float | None, float | None, str | None]:
        valid = [
            (
                _to_float(row.get("timestamp_unix_s")),
                _to_float(row.get("power_draw_w")),
            )
            for row in self._sample_rows
        ]
        valid = [(ts, power) for ts, power in valid if ts is not None and power is not None]
        if len(valid) < 2:
            return None, None, "fewer than two valid timestamped power samples"
        energy_j = 0.0
        duration_s = 0.0
        for (t0, p0), (t1, p1) in zip(valid, valid[1:]):
            dt = max(0.0, t1 - t0)
            energy_j += 0.5 * (p0 + p1) * dt
            duration_s += dt
        if duration_s <= 0:
            return None, None, "valid samples did not span a positive duration"
        return energy_j, energy_j / duration_s, None

    def _cap_saturation(self) -> tuple[float | None, str | None, int]:
        if self.actual_power_limit_w is None or self.actual_power_limit_w <= 0:
            return None, "verified actual_power_limit_w is unavailable", 0
        powers = [
            power
            for power in (_to_float(row.get("power_draw_w")) for row in self._sample_rows)
            if power is not None
        ]
        if not powers:
            return None, "no valid observed power samples", 0
        saturated = sum(
            power / self.actual_power_limit_w >= 0.95 for power in powers
        )
        return saturated / len(powers), None, len(powers)

    def finalize(self, *, workload_exception: BaseException | None = None) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("HardwareProfileRecorder must be started before finalize()")
        if self._finalized:
            assert self.summary is not None
            return self.summary
        self._finalized = True

        # ``__exit__`` is entered immediately after the measured workload, and
        # the synchronous endpoint sample below therefore follows it.
        workload_end_s = self._clock()
        self._stop_event.set()
        if self._sample_thread is not None:
            self._sample_thread.join(timeout=max(1.0, self.sample_interval_s * 2.0))
        # A synchronous endpoint sample brackets even very short workloads.
        self._collect_sample()

        if self._samples_file is not None:
            self._samples_file.flush()
            self._samples_file.close()
            self._samples_file = None
            self._samples_writer = None

        energy_end = self._query_energy_counter()
        energy_start = (
            self._after_snapshot.get("energy_counter", {})
            if self._after_snapshot is not None
            else {}
        )
        counter_start_mj = _to_float(energy_start.get("total_energy_mj"))
        counter_end_mj = _to_float(energy_end.get("total_energy_mj"))
        energy_j: float | None = None
        energy_missing_reason: str | None = None
        counter_missing_reason: str | None = None
        if (
            counter_start_mj is not None
            and counter_end_mj is not None
            and counter_end_mj >= counter_start_mj
        ):
            energy_j = (counter_end_mj - counter_start_mj) / 1000.0
            self.energy_measurement_source = "nvidia_smi_total_energy_counter"
        else:
            counter_missing_reason = (
                energy_start.get("missing_reason")
                or energy_end.get("missing_reason")
                or "energy counter decreased during the workload"
            )
            sampled_energy_j, _sampled_avg, sampled_reason = self._sample_energy()
            if sampled_energy_j is not None:
                energy_j = sampled_energy_j
                self.energy_measurement_source = "nvidia_smi_power_sampling_trapezoidal"
            else:
                self.energy_measurement_source = None
                energy_missing_reason = (
                    f"energy counter unavailable ({counter_missing_reason}); "
                    f"power-sampling fallback unavailable ({sampled_reason})"
                )

        sampled_energy_j, sampled_avg_power_w, sampled_energy_reason = self._sample_energy()
        self.cap_saturated_ratio, cap_reason, valid_power_samples = self._cap_saturation()
        valid_timestamps = [
            ts
            for ts in (_to_float(row.get("timestamp_unix_s")) for row in self._sample_rows)
            if ts is not None
        ]
        nvml_supported = bool(energy_start.get("supported") and energy_end.get("supported"))
        identity = (
            self._after_snapshot.get("gpu_identity", {})
            if self._after_snapshot is not None
            else {}
        )
        self.summary = {
            "schema_version": "april.hardware_summary.v1",
            "requested_profile": self.requested_profile,
            "profile_status": self.profile_status,
            "actual_power_limit_w": self.actual_power_limit_w,
            "actual_power_limit_missing_reason": (
                self._after_snapshot.get("actual_power_limit_missing_reason")
                if self._after_snapshot is not None
                else "after-apply snapshot unavailable"
            ),
            "energy_measurement_source": self.energy_measurement_source,
            "energy_measurement_missing_reason": energy_missing_reason,
            "energy_j": energy_j,
            "nvml_energy_counter_supported": nvml_supported,
            "nvml_energy_counter_missing_reason": counter_missing_reason,
            "energy_counter_start_mj": counter_start_mj,
            "energy_counter_end_mj": counter_end_mj,
            "energy_counter_end_query": energy_end["query"],
            "sampled_energy_j": sampled_energy_j,
            "sampled_avg_power_w": sampled_avg_power_w,
            "sampled_energy_missing_reason": sampled_energy_reason,
            "cap_saturated_ratio": self.cap_saturated_ratio,
            "cap_saturated_threshold_fraction": 0.95,
            "cap_saturated_detector": "fraction_of_valid_samples_at_or_above_0.95_actual_limit",
            "cap_saturated_missing_reason": cap_reason,
            "sample_count": len(self._sample_rows),
            "valid_power_sample_count": valid_power_samples,
            "sample_interval_s": self.sample_interval_s,
            "workload_start_unix_s": self._workload_start_timestamp_s,
            "workload_end_unix_s": workload_end_s,
            "first_sample_unix_s": min(valid_timestamps) if valid_timestamps else None,
            "last_sample_unix_s": max(valid_timestamps) if valid_timestamps else None,
            "samples_bracket_workload": bool(
                valid_timestamps
                and self._workload_start_timestamp_s is not None
                and min(valid_timestamps) <= self._workload_start_timestamp_s
                and max(valid_timestamps) >= workload_end_s
            ),
            "gpu_selector": self.gpu_id,
            "gpu_index": identity.get("index"),
            "gpu_uuid": identity.get("uuid"),
            "gpu_name": identity.get("name"),
            "profile_verification_query": (
                self._after_snapshot.get("verification_query")
                if self._after_snapshot is not None
                else None
            ),
            "workload_exception": (
                {
                    "type": type(workload_exception).__name__,
                    "message": str(workload_exception),
                }
                if workload_exception is not None
                else None
            ),
            "artifacts": {
                "snapshot_before": self.snapshot_before_path.name,
                "snapshot_after_apply": self.snapshot_after_path.name,
                "samples": self.samples_path.name,
                "summary": self.summary_path.name,
            },
        }
        self._write_json(self.summary_path, self.summary)
        return self.summary

    def metadata(self) -> dict[str, Any]:
        """Return the five required fields for merging into result rows."""

        source: Mapping[str, Any]
        if self.summary is not None:
            source = self.summary
        else:
            source = {
                "requested_profile": self.requested_profile,
                "profile_status": self.profile_status,
                "actual_power_limit_w": self.actual_power_limit_w,
                "energy_measurement_source": self.energy_measurement_source,
                "cap_saturated_ratio": self.cap_saturated_ratio,
            }
        return {field: source.get(field) for field in REQUIRED_FIELDS}

    def merge_metadata(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Copy a result row and overwrite hardware fields with observed values."""

        merged: MutableMapping[str, Any] = dict(row)
        merged.update(self.metadata())
        return dict(merged)
