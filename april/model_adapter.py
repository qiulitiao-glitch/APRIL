"""APRIL-owned API glue; all Transformer block operations stay in external APIs."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

_upstream = None


def configure_layerskip(root):
    global _upstream
    root = Path(root).resolve()
    package = Path(__file__).resolve().parents[1]
    if root == package or package in root.parents:
        raise ValueError('LayerSkip must remain outside the public repository')
    pins = json.loads((package/'configs/external_sources.json').read_text())['LayerSkip']
    for name, digest in pins['sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Official LayerSkip dependency identity mismatch: '+name)
    name = '_april_official_layerskip_model_api'
    spec = importlib.util.spec_from_file_location(name, root/'self_speculation/llama_model_utils.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _upstream = module
    return pins['revision']


def forward(model, input_ids, past_key_values):
    if _upstream is None:
        raise RuntimeError('Configure a verified official LayerSkip checkout first')
    return _upstream.forward(model, input_ids, past_key_values)


def shallow_forward(model, input_ids, past_key_values, exit_layer, exit_query_cache=None):
    if _upstream is None:
        raise RuntimeError('Official LayerSkip has not been configured')
    return _upstream.forward_early(model, input_ids, past_key_values, exit_layer, exit_query_cache)


def crop_past_key_values(cache, maximum_length):
    """Keep committed sequence positions using ordinary tensor slicing."""
    if maximum_length < 0:
        raise ValueError('Negative committed cache length')
    if cache is None:
        return None
    return tuple(None if pair is None else tuple(
        None if tensor is None else tensor[..., :maximum_length, :]
        for tensor in pair) for pair in cache)


def configure_torch():
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def load_model(path, layerskip_root):
    import torch
    import transformers
    if transformers.__version__ != '4.45.2' or torch.__version__ != '2.2.1+cu121':
        raise RuntimeError('Use the frozen Transformers 4.45.2 / torch 2.2.1+cu121 environment')
    configure_torch()
    configure_layerskip(layerskip_root)
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError('Model path must be a locally acquired checkpoint')
    from .config import paper_snapshot
    identity=paper_snapshot()['sections']['software_environment']['model_identity']['files']
    for entry in identity:
        file=path/entry['relative_path']
        if not file.is_file() or file.stat().st_size!=entry['size_bytes']:
            raise ValueError('Frozen model/tokenizer file missing or wrong size: '+entry['relative_path'])
        digest=hashlib.sha256()
        with file.open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=entry['sha256']:
            raise ValueError('Frozen model/tokenizer hash mismatch: '+entry['relative_path'])
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        path, use_fast=True, trust_remote_code=False, local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        trust_remote_code=False, local_files_only=True, use_safetensors=True,
        attn_implementation='eager').cuda().eval()
    if len(model.model.layers) != 32:
        raise ValueError('This frozen APRIL profile requires a 32-block model')
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer
