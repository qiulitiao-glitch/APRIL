# Dataset identity

Only IDs, request order and SHA256 hashes are shipped. Public acquisition revisions
in `download_sources.json` identify content candidates, not unrecorded historical
download revisions. Preparation checks every prompt and reference and fails before
writing final output if an ID or hash differs. No generated HumanEval code runs.

| Task | Test source and selection | Requests |
|---|---|---:|
| CNN/DM summarization | abisee/cnn_dailymail 3.0.0, frozen IDs | 1,000 |
| CNN/DM continuation | Same selected articles; word-based construction/filter | 998 |
| XSum | EdinburghNLP/xsum, frozen IDs | 1,000 |
| HumanEval | openai/openai_humaneval, all problems | 164 |
| WMT14 de-en | sacreBLEU wmt14/full, first 1,000 pairs | 1,000 |
| GSM8K | openai/gsm8k main, sorted Random(42) sample from 1,319 | 1,000 |
| AQuA-RAT | google-deepmind/AQuA test.json, all examples | 254 |

`prepare_data.py` contains the literal prompt templates. It does not deduplicate
text under distinct IDs. CNN/DM continuation skips articles shorter than 80 words,
takes `min(160,max(64,len(words)//3))` input words, and uses up to the next 96
words as the reference. Generation does not request tokenizer truncation.

Existing raw files can replace downloads. The required names are
`cnn_dm_test.parquet`, `xsum_test.parquet`, `gsm8k_test.parquet`,
`humaneval_test.parquet`, `aqua_test.jsonl`, `wmt14_de.txt`, `wmt14_en.txt`.
The four parquet sources also accept JSONL with their original columns. Retain
the full 1,319-row GSM8K raw split. Prepared prompts are private local artifacts.

Run `split_requests.py` after preparation to preserve the two independent main
controller streams and their recorded task/request ordering.
