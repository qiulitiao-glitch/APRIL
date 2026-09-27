# Third-party boundaries

No third-party implementation, model weights or benchmark payloads are vendored.

| Component | How obtained | Terms / identity |
|---|---|---|
| LayerSkip | Separate official checkout; required by public APRIL | [CC BY-NC 4.0 at pinned revision](https://github.com/facebookresearch/LayerSkip/blob/494752e5fbb0a82989f6cb384841684b1c2ef5c3/LICENSE), revision `494752e5fbb0a82989f6cb384841684b1c2ef5c3`; runtime checks source and license hashes |
| LayerSkip Llama-2 7B checkpoint | Separate authorized model acquisition | [Model repository](https://huggingface.co/facebook/layerskip-llama2-7B); model terms are separate from source-code terms |
| PyTorch / Transformers | Installed packages | [PyTorch license](https://github.com/pytorch/pytorch/blob/v2.2.1/LICENSE), [Transformers license](https://github.com/huggingface/transformers/blob/v4.45.2/LICENSE) |
| DEL | External historical baseline only | [Official repository](https://github.com/hoenza/DEL), recorded comparison revision `6369bf7cc0304e1aa474af65e7a0675f2663c31c`; no DEL source distributed or imported |

DEL citation: H. Entezari Zarch, L. Gao, C. Jiang, and M. Annavaram,
“DEL: Context-aware dynamic exit layer for efficient self-speculative decoding,”
COLM 2025, [paper](https://arxiv.org/abs/2504.05598).
Current upstream DEL is not asserted to reproduce the paper's locally modified
historical baseline. Archived author-generated numeric measurements suffice for
Table 2 regeneration without DEL source.

The LayerSkip public dependency pin is not a claim that the original experiments
recorded that upstream Git revision. File identities, experiment settings and
validation results provide the reproduction evidence. APRIL-owned source code is licensed under PolyForm Noncommercial 1.0.0,
Copyright 2026 Litiao Qiu; see [LICENSE](LICENSE) and [NOTICE](NOTICE). This
license does not relicense or override any third-party software, model or dataset.
