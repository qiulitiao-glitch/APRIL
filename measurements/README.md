# Archived measurements

Table 2 contains 27,080 numeric records: five methods on the same 5,416 request
identities. Table 4 contains 1,792 records: four configurations on 448 paired
requests. No prompt text or generated benchmark output is distributed.

Regeneration verifies file hashes, unique request/method identity, complete
populations, prompt hashes, token counts and positive finite elapsed times. TPS is
`sum(committed_tokens)/sum(generation_seconds)`, within task or across all tasks.
It never averages per-request TPS or consumes printed table cells as measurements.
EOS-shortened requests retain their actual output length and measured duration.

PLD's archived three AR token mismatches are retained as diagnostics. The paper
does not claim universal losslessness; mismatch alone does not invalidate this
measurement. The original archival status remains visible in evidence metadata.

Regenerating tables is distinct from rerunning methods. DEL is unnecessary for
regeneration. The archive does not establish any unrecovered timing CI procedure;
no timing confidence interval is fabricated by these scripts.
