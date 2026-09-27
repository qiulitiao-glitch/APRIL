# Reproduction environment

Use Linux x86_64, Python 3.10, a CUDA 12.1-compatible NVIDIA driver, and an RTX
6000 Ada GPU for the main-profile matched validation. PyTorch must report
`2.2.1+cu121`; Transformers must report `4.45.2`. The runtime checks these versions.

Install PyTorch from its CUDA 12.1 wheel index before editable installation.
`pyproject.toml` pins the public validation package versions. Historical main
environment variables beyond the recovered deterministic flags were not all
recorded; no completeness claim is made for unavailable incidental metadata.

Obtain the model through its authorized Hugging Face access flow. The acquisition
revision `7e0b20094ef76fb18fd8a9bdf7ee019c3d7b8880` has recorded file identities;
it is not a fabricated original experiment revision. Runtime loading requires the
recorded file sizes and hashes, local safetensors, and `trust_remote_code=False`.

Obtain official LayerSkip separately with `setup_layerskip.py`. The adapter loads
only the hash-checked official model API module. No DEL or mixed historical helper
is installed by these commands. This source checkout requires editable installation
because the frozen metadata remain adjacent to the Python package.

Hardware recording samples power and process occupancy; it does not impose power
or clock limits. External compute or failed occupancy observations invalidate a run.
Sampling does not provide continuous operating-system process tracing.
