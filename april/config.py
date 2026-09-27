"""Read the authoritative snapshot; unknown historical rules are never defaults."""
from dataclasses import dataclass
import json
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / 'configs/paper/PAPER_FROZEN_CONFIG.json'


def paper_snapshot():
    return json.loads(SNAPSHOT_PATH.read_text(encoding='utf-8'))


def require_frozen_config(path):
    """An explicit command argument must select the installed frozen profile."""
    import hashlib
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(SNAPSHOT_PATH.read_bytes()).digest():
        raise ValueError('Configuration differs from the frozen paper snapshot')
    return json.loads(path.read_text(encoding='utf-8'))


PAPER = paper_snapshot()['sections']['april_main']

LAYERSKIP_REVISION = '494752e5fbb0a82989f6cb384841684b1c2ef5c3'
EXIT_LAYERS = tuple(PAPER['E'])
DRAFT_LENGTHS = tuple(PAPER['K'])


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = PAPER['generation']['max_new_tokens']
    margin_threshold: float = PAPER['verification']['margin']
    draft_confidence_threshold: float = PAPER['shallow_stopping']['tr_nes_confidence_threshold']
    draft_confidence_stop_after: int = PAPER['shallow_stopping']['tr_nes_confidence_stop_after']
    min_ngram: int = PAPER['lookup']['min_ngram']
    max_ngram: int = PAPER['lookup']['max_ngram']
    max_candidates: int = PAPER['lookup']['max_candidates']

    def __post_init__(self):
        if not 1 <= self.max_new_tokens <= 256:
            raise ValueError('The paper profile supports 1..256 output tokens')
        if self.margin_threshold != 0.02:
            raise ValueError('The frozen margin threshold is 0.02')
