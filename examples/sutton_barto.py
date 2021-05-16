"""
Multi-armed Bandit examples taken from Reinforcement Learning: An Introduction
by Sutton and Barto, 2nd ed. rev Oct2015.
"""
import sys
# sys.path.append("./bandits")
from bandits.environment import Environment
from bandits.bandit import GaussianBandit
from bandits.agent import Agent, GradientAgent
from bandits.policy import (EpsilonGreedyPolicy, GreedyPolicy, UCBPolicy,
                            SoftmaxPolicy)


class EpsilonGreedyExample:
    label = 'Action-Value Methods'
    bandit = GaussianBandit(10)
    GreedyPolicy(1)
    agents = [
        Agent(bandit, GreedyPolicy(1)),
        Agent(bandit, EpsilonGreedyPolicy(0.01, 1)),
        Agent(bandit, EpsilonGreedyPolicy(0.1, 1)),
        # Agent(bandit, EpsilonGreedyPolicy(0.1, 1), prior=5),
        # Agent(bandit, GreedyPolicy(1), prior=5)
        # Agent(bandit, GreedyPolicy(10)),
        # Agent(bandit, EpsilonGreedyPolicy(0.1, 10)),
    ]


class OptimisticInitialValueExample:
    label = 'Optimistic Initial Values'
    bandit = GaussianBandit(10)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1, 1)),
        Agent(bandit, GreedyPolicy(1)),
        Agent(bandit, GreedyPolicy(1), prior=1),
        Agent(bandit, GreedyPolicy(1), prior=2),
        Agent(bandit, GreedyPolicy(1), prior=5),
        Agent(bandit, GreedyPolicy(1), prior=10),
    ]


class UCBExample:
    label = '2.6 - Upper-Confidence-Bound Action Selection'
    bandit = GaussianBandit(10)
    agents = [
        Agent(bandit, GreedyPolicy(1)),
        Agent(bandit, GreedyPolicy(1), prior=10),
        Agent(bandit, EpsilonGreedyPolicy(0.1, 1)),
        Agent(bandit, UCBPolicy(2)),
    ]


class GradientExample:
    label = '2.7 - Gradient Bandits'
    bandit = GaussianBandit(10, mu=4)
    policy = SoftmaxPolicy()
    agents = [
        GradientAgent(bandit, policy, alpha=0.1),
        GradientAgent(bandit, policy, alpha=0.4),
        GradientAgent(bandit, policy, alpha=0.1, baseline=False),
        GradientAgent(bandit, policy, alpha=0.4, baseline=False)
    ]


if __name__ == '__main__':
    experiments = 500
    trials = 1000

    # example = EpsilonGreedyExample
    # example = OptimisticInitialValueExample
    # example = UCBExample
    example = GradientExample

    env = Environment(example.bandit, example.agents, example.label)
    scores, optimal = env.run(trials, experiments)
    env.plot_results(scores, optimal)
