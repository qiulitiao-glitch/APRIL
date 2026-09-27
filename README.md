# APRIL

Reproduction candidate for **APRIL: Adaptive Proposal Routing with In-Request Lookup
for Self-Speculative LLM Serving at the Edge**. APRIL tries request-local lookup
before adaptive shallow drafting and verifies proposals with the full model.

This repository provides the APRIL implementation, frozen experimental
configurations, and scripts for reproducing the main results reported in the
paper. It focuses on the main throughput evaluation (Table 2) and core ablation
(Table 4). Published tables are reconstructed from archived per-request
measurements; the APRIL runtime and public baselines can also be run on newly
prepared inputs. Auxiliary diagnostics and the historical DEL and ablation
orchestration are outside the rerunnable scope of this package.

**Historical technical qualification passed:** 448/448 public/historical outputs match,
with no measured mechanism drift and +0.26% ratio-of-sums TPS difference.
The current public snapshot normalizes descriptive labels and stream names.
Its executable algorithm and numeric settings are unchanged; source checks and
repeat smoke evidence are recorded in
[metadata validation](measurements/public_metadata_validation.json).
APRIL-owned source code is licensed under [PolyForm Noncommercial 1.0.0](LICENSE).
Copyright 2026 Litiao Qiu. See [NOTICE](NOTICE) for scope and required attribution.

The repository contains original APRIL components, independently assembled runtime
integration, frozen configurations, request hashes, and numeric measurement archives.
Model weights, benchmark text, DEL source and third-party framework code are acquired
separately. The public implementation makes no universal losslessness claim.

## Install

Use Linux, Python 3.10 and an idle RTX 6000 Ada for the main profile. A model
checkpoint and the pinned CUDA stack are required for generation; table regeneration
needs no model/GPU. Start from this source directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[test,data]'
export CONFIG="$PWD/configs/paper/PAPER_FROZEN_CONFIG.json"
export WORK="$PWD/data_work"
export MODEL="$WORK/model"
export LAYERSKIP="$(dirname "$PWD")/LayerSkip-external"
export GPU=0
mkdir -p "$WORK" results
python scripts/setup_layerskip.py --config "$CONFIG" --destination "$LAYERSKIP"
```

`GPU` is the physical index of an idle RTX 6000 Ada on your machine. The CLI refuses
an occupied GPU and invalidates results when sampled external compute is detected.
See [dependencies](docs/DEPENDENCIES.md) for package and model identities.

## Obtain model and data

Accept the [model access terms](https://huggingface.co/facebook/layerskip-llama2-7B)
using your own account, then acquire the recorded public revision:

```bash
huggingface-cli login
huggingface-cli download facebook/layerskip-llama2-7B \
  --revision 7e0b20094ef76fb18fd8a9bdf7ee019c3d7b8880 --local-dir "$MODEL"
python scripts/download_data.py --config "$CONFIG" --output "$WORK/raw" --dry-run
python scripts/download_data.py --config "$CONFIG" --output "$WORK/raw"
python scripts/prepare_data.py --config "$CONFIG" \
  --raw-root "$WORK/raw" --output "$WORK/prepared"
python scripts/split_requests.py --config "$CONFIG" \
  --input "$WORK/prepared/inputs.jsonl" --output "$WORK/streams"
```

Existing raw files can replace acquisition; required filenames are in
[data/README.md](data/README.md). Preparation checks all 5,416 frozen prompt/reference
hashes and fails closed on changed data. It never executes HumanEval code.
`stream1.jsonl` and `stream2.jsonl` identify independent controller streams.
Run both on the selected idle device, preserving separate controller lifetimes.
Qualification reused existing raw data and verified every frozen identity. The
download manifest and dry-run were checked; a complete fresh network download was
not qualified. Public source availability and access depend on the upstream hosts.

## Smoke test and tests

```bash
pytest -q
python scripts/audit_frozen_config.py --config "$CONFIG" --output results/config_audit.json
python scripts/smoke_test.py --config "$CONFIG" --model "$MODEL" \
  --layerskip "$LAYERSKIP" --gpu "$GPU" --output results/smoke
python scripts/aggregate.py --config "$CONFIG" --run results/smoke \
  --output results/smoke_aggregate.json
```

Smoke uses two original synthetic prompts with a 32-token cap. Expected behavior:
64 generated tokens, lookup hits and lookup-miss self-drafts, verification, commits,
controller updates, and a complete timing/hardware record. Timing varies by machine.
Every output path must be new; commands avoid overwriting prior measurements.

## Run APRIL and public baselines

The following commands run full paper request populations and can take hours.
For a small diagnostic, append `--limit 9`; it is not a full-population result.
Archived generation times total about 9.5 GPU-hours for APRIL alone and 51.8
GPU-hours for all four public methods, before setup and data preparation. The
commands below execute the two streams sequentially on one selected GPU. No full
5,416-request inference campaign was launched during release qualification.

```bash
for STREAM in stream1 stream2; do
  python scripts/run_april.py --config "$CONFIG" --model "$MODEL" \
    --layerskip "$LAYERSKIP" --gpu "$GPU" --input "$WORK/streams/$STREAM.jsonl" \
    --output "results/april_$STREAM"
  python scripts/run_ar.py --config "$CONFIG" --model "$MODEL" \
    --layerskip "$LAYERSKIP" --gpu "$GPU" --input "$WORK/streams/$STREAM.jsonl" \
    --output "results/ar_$STREAM"
  python scripts/run_layerskip.py --config "$CONFIG" --model "$MODEL" \
    --layerskip "$LAYERSKIP" --gpu "$GPU" --input "$WORK/streams/$STREAM.jsonl" \
    --output "results/layerskip_$STREAM"
  python scripts/run_pld.py --config "$CONFIG" --model "$MODEL" \
    --layerskip "$LAYERSKIP" --gpu "$GPU" --input "$WORK/streams/$STREAM.jsonl" \
    --output "results/pld_$STREAM"
done
python scripts/aggregate.py --config "$CONFIG" \
  --run results/april_stream1 results/april_stream2 results/ar_stream1 results/ar_stream2 \
        results/layerskip_stream1 results/layerskip_stream2 results/pld_stream1 results/pld_stream2 \
  --output results/rerun_tps.json
```

DEL remains an [external historical baseline](https://github.com/hoenza/DEL).
The frozen experiment-level profile and archived author measurements are included;
current upstream DEL is not claimed to reproduce the locally modified paper arm.

## Regenerate published tables

These commands use archived per-request counts and elapsed times, validate complete
populations and identities, and do not launch benchmark inference:

```bash
python scripts/regenerate_table2.py --config "$CONFIG" --output results/table2
python scripts/regenerate_table4.py --config "$CONFIG" --output results/table4
```

Table 2 overall TPS rounds to AR 20.62, LayerSkip 28.04, DEL 31.53, PLD 37.25,
APRIL 40.04. Table 4 rounds to static-only 31.48, adaptive-only 32.08,
static+lookup 39.45, APRIL 40.00. Table 4 reconstruction is supported; rerunning
all historical ablation orchestration is outside this minimal package.

## Timing, configuration and outputs

TPS is total committed output tokens divided by total measured generation seconds.
CUDA synchronization brackets generation, including prefill, lookup, controller,
self-draft, target verification, cache management and numerical recovery. Loading,
warmup, tokenization, detokenization, hardware boundary queries and output writes
are excluded. Recovery is not always one extra forward. No timing CI is invented.

All runtime knobs are specified in the [frozen snapshot](configs/paper/PAPER_FROZEN_CONFIG.json)
and [hyperparameter guide](docs/PAPER_HYPERPARAMETERS.md). Main E={7,8},
K={4,6,8,10,12,16,20}; historical A800/H100 settings are documented as metadata.
The CLI currently validates the main profile only. Global controller statistics
persist within a stream, while fast request state resets.

Each generation output contains `run_manifest.json`, `results.jsonl`, `summary.json`
and `hardware/` snapshots, samples and summary. Generated token IDs and local
machine metadata are private run artifacts and must not be added to the source tree.

Historical model/dataset download revisions and parts of original portfolio-selection
provenance were not recorded. Content hashes establish artifact identity. The
portfolio is a frozen exploratory selection; the disjoint calibration scan is
validation-only. PLD's three archived AR token mismatches alone do not invalidate
its measurements under the paper's non-lossless claim.

See [reproducibility](docs/REPRODUCIBILITY.md), [third-party notices](THIRD_PARTY_NOTICES.md)
and [citation metadata](CITATION.cff). The APRIL license applies only to APRIL-owned source; third-party materials
retain their own terms. No DOI is included because none has been supplied.
