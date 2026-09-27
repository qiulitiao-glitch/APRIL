# Frozen paper configuration

`configs/paper/PAPER_FROZEN_CONFIG.json` is authoritative. The eight YAML files
are checked projections. Generation commands require its explicit path and reject
a different configuration. Full factory fields, controller constructor values,
initial state, inactive switches, model/tokenizer hashes, dataset identity and
cross-GPU projections are preserved in this snapshot.

| Active setting | Value |
|---|---|
| Checkpoint | facebook/layerskip-llama2-7B, 32 blocks |
| Precision / attention / cache | FP16 model, logits and KV; eager; DynamicCache through official LayerSkip APIs |
| Decode | Greedy, batch 1, concurrency 1 per controller stream, max 256 new tokens |
| Stop / tokenize | Stop at EOS and omit returned EOS; add_special_tokens=True; no batch padding or tokenizer truncation |
| RNG | torch seed 20260712 + source_index |
| Main actions | E=[7,8], K=[4,6,8,10,12,16,20], K-major then ascending E |
| Transfer actions | E=[7,8], K=[4,12,20] on A800/H100; no retuning claimed |
| LayerSkip baseline | E=7, K=6, draft stops at EOS; no confidence early stop |
| PLD baseline | Transformers 4.45.2 candidate generator, n<=2, up to 10 proposal tokens, AR on miss |
| Verification guard | Commit-critical min top1-top2 logit margin <=0.02 |

Lookup tests key lengths 4,3,2 and continuation lengths 8,6,4,2. Backward context
matching stops at length 32. The bound is 64 eligible exact-key matches per pass;
skipped bucket positions do not count. Hash base is 1,000,003, arithmetic is modulo
2^64, token offset is one, and every key hash match is checked against exact tokens.
The prompt and target-committed tokens populate a request-local index. Each new
token contributes at most three entries; capacity is prompt length plus output cap.
Pinned host and device int64 candidate buffers each have shape [1,8].

| Proposal length | Minimum context | Support condition | Minimum consistency |
|---:|---:|---|---:|
| 8 | 4 | >=2 | .75 |
| 6 | 3 | >=2 or matching context >=6 | .67 |
| 4 | 3 | positive support through consistency test | .67 |
| 2 | 2 | >=1 | .50 |

Selection groups matches by next token, choosing lexicographically by count,
matching context, available continuation and occurrence position. The representative
occurrence maximizes context, continuation and position. The first qualifying
continuation/key wins. The output cap reserves one correction/bonus position.

The controller uses EMA old-value weight .7 (new-observation coefficient beta=.3),
conservative adjustment lambda=.5, minimum
observations=1 drafted token, and context buckets of 128 tokens. Global/context
rates use EMA; recent rates use accepted-token sum / drafted-token sum over eight
rounds. The conservative estimate takes the minimum of available views. Ordinary
feedback maps acceptance below .70 to K=4, [.70,.88) to K=12 and >=.88 to K=20,
subject to probe/cooldown rules. The active layer switching penalty is .08;
the separate .04 measured-utility penalty is inactive because runtime timing
observations are disabled. Exact clamps, scoring, tie rules, cooldown/probe state
and initial counters are recorded under the controller fields in the snapshot.

Confidence is the greedy shallow token's softmax probability from unscaled FP16
logits. At each self-draft round, start at .55; add .05 if bad_streak>0, then add
.025 if cooldown>0, then subtract .025 if good_streak>=2, and clamp to [.35,.70].
Conditions are independent. Append a draft token, stop on EOS first, then stop if
its probability is below threshold after at least one draft token. The threshold
has no additional layer or position dependence. Adaptive relaxation is disabled.

Request reset clears fast context/recent state while retaining global aggregate
acceptance within a controller stream. Main paper evaluation uses two streams;
starting a fresh process for every request changes the experiment. Measured-action
utility statistics stay empty when latency observations are disabled.

The numerical guard can prefill a canonical checkpoint, replay a committed suffix
using Q1 steps, recheck candidates and advance the correction checkpoint. Its cost
is inside generation time; it is not always one additional forward. This mechanism
does not imply universal AR equality.

The retained action set was hand-set from exploratory profiling. Recovered
evaluation-disjoint 64-prompt calibration evidence is validation-only; it is not
retroactively described as the selector. Unrecorded historical download revisions
and original selection details are classified separately from runtime parameters.

Supplemental timing intervals use 10,000 paired-request bootstrap replicates and
95% percentile intervals, with linear interpolation at `(N-1)*q`. Overall resampling
preserves workload counts. A800/H100 use `random.Random(20260716)` overall; exact
per-workload and moving-block seeds are recorded in the snapshot. Output-budget
analysis uses limits 64/128/256 and seed `20260715 + 100*budget + workload_index`,
with overall at index zero and the exact historical workload order recorded there.
These are recovered supplemental procedures; they do not establish a main Table 2
timing-CI procedure. Table 2 regeneration emits point estimates only.
