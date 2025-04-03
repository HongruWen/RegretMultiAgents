"""
Baseline implementations for regret calculation.
"""

from regrets.baselines.base import BaseBaseline
from regrets.baselines.rl_policy import RLPolicyBaseline

__all__ = ['BaseBaseline', 'RLPolicyBaseline'] 