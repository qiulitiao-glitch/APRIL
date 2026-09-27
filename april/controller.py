"""Construct the audited APRIL-owned controller with the exact frozen profile."""
from .controller_core import TargetRelativeNestedExitController
from .config import paper_snapshot


def create_controller():
    parameters = paper_snapshot()['sections']['april_main']['controller_parameters']
    if parameters['anchor_candidates'] != [7, 8] or parameters['action_k_candidates'] != [4, 6, 8, 10, 12, 16, 20]:
        raise ValueError('Frozen action set mismatch')
    return TargetRelativeNestedExitController(**parameters)
