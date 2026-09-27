# Reproducibility scope

The frozen snapshot is authoritative. Readable YAML files are exact projections;
the configuration audit checks their agreement and the source evidence. Public
commands require `--config` and reject a modified snapshot supplied at another path.
Changing the canonical file itself creates a different experiment and requires a
new configuration audit and implementation qualification.

## Evidence supported by this archive

Table 2 contains 5,416 requests for each of five methods. Table 4 contains the same
448 paired request identities for each of four ablation arms. Counts and timing
are retained per request; regeneration uses total tokens divided by total seconds,
not the mean of request-level throughput. Regeneration does not run inference.

The data preparation command reconstructs all 5,416 prompt and reference identities
from separately obtained original data. It refuses altered content. Source IDs,
request order, construction rules and hashes are included; benchmark text is not.
Model loading similarly verifies the checkpoint and tokenizer file identities.

The main profile uses FP16, eager attention, batch size one, greedy decoding and
a 256-token output cap. Each historical request stream starts with a fresh
controller. Global acceptance statistics persist within that stream; request-local
statistics reset. A prefix diagnostic and a resumed campaign are different controller
histories. This package does not offer qualified mid-stream resume.

## Runtime qualification

Qualification passed on the main RTX 6000 Ada profile. A two-prompt real GPU
smoke in a fresh virtual environment exercised lookup hits, self-drafting,
verification and controller updates, with only this repository and documented
external dependencies accessible. AR, LayerSkip and PLD also passed their small
canaries and aggregation. No historical or DEL source was accessible to these runs.

The matched 64-request trial produced 64/64 equal outputs and
+0.34% public/historical ratio-of-sums TPS difference. The final
448-request trial (64 per workload) produced 448/448 equal outputs, identical
measured routing, actions, draft acceptance and recovery counts, no extra normal
target forwards, and +0.26% TPS difference. Both paths used the same
explicitly identified physical GPUs and timing boundary. The original threshold
was ±3%; it was not relaxed. See the [numeric qualification record](../measurements/implementation_validation.json).
These observations do not establish universal equality with autoregressive decoding.

The shipped generation source equals the validated source byte for byte. New
user runs keep `paper_reproduction_qualified=false` in their manifests: arbitrary
inputs or partial runs do not automatically qualify as full paper reproductions.

Timing includes CUDA-synchronized generation, prefill, proposals, verification,
cache operations, recovery and Python control flow. Loading, warmup, tokenization,
detokenization, record I/O and boundary hardware queries are excluded. Background
hardware sampling runs throughout the process; its run-level energy scope includes
setup and warmup and must not be interpreted as generation-only energy. No power
limit is changed by default.

## Known provenance limits

The archived content hashes establish model and data identity even where the
historical download revision was not recorded. The published acquisition revision
is an independently verified way to obtain those bytes, not a recovered historical
download log. The retained E/K action set is known; the disjoint 64-prompt calibration
scan is validation-only and does not establish its original selection procedure.
Main Table 2 throughput has no reconstructed timing confidence interval; none is
invented here. The classified unknowns are retained in the frozen snapshot.

The three archived PLD-versus-AR token mismatches do not invalidate the measured
PLD throughput under the paper's non-lossless claim. DEL measurements are archived
author results. Its locally modified historical implementation is not redistributed
or claimed reproducible by running current upstream DEL.

Install verification and runtime validation apply to the recorded package versions.
Cross-device settings are documented, but the public CLI currently implements the
main RTX 6000 Ada profile. Device-specific throughput requires a new measurement.

APRIL-owned source code is licensed under PolyForm Noncommercial 1.0.0,
Copyright 2026 Litiao Qiu. Third-party terms remain separate. Citation metadata
uses the author order in the available ICSOC 2026 camera-ready paper; no DOI or
ORCID is inferred.
