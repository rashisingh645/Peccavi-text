"""
peccavi/auctor_synthid.py
Baseline: SynthID-Text (Dathathri et al., 2024, Google DeepMind)
Tournament sampling watermark with a fixed strength parameter — no adaptive policy learning.

This is structurally identical to PECCAVI's Auctor with adaptive_theta=False.
It establishes the "SynthID baseline" row in the paper: the value of PECCAVI's
REINFORCE-learned adaptive theta is measured against this fixed-theta version.
Detection goes through Custos (same hash-based partition as PECCAVI).
"""

from __future__ import annotations
from backbone.model import LLaMABackbone
from peccavi.auctor import Auctor
from peccavi.constants import SECRET_KEY


class SynthIDAuctor:
    """
    SynthID-Text: fixed-theta tournament sampling with no policy gradient updates.
    Wraps PECCAVI's Auctor; detection is handled by the shared Custos scorer.
    """

    def __init__(self, backbone: LLaMABackbone, theta: float = 2.0,
                 tournament_k: int = 8, secret_key: str = SECRET_KEY):
        self._auctor = Auctor(backbone, theta=theta,
                              tournament_k=tournament_k, secret_key=secret_key)
        self.backbone = backbone
        self.theta = theta

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        return self._auctor.generate(prompt, max_tokens=max_tokens)
