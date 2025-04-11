"""
Baseline implementations for regret calculation.
"""

from regrets.baselines.base import BaseBaseline
from regrets.baselines.rl_policy import RLPolicyBaseline
from regrets.baselines.greedy_chaser import GreedyTargetChaser

__all__ = ['BaseBaseline', 'RLPolicyBaseline', 'GreedyTargetChaser'] 