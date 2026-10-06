import random
from math import isclose

from gambler_problem.env import GamblingGame


class GreedyPolicy:
    def __init__(
        self,
        env: GamblingGame,
        action_map: dict[int, tuple[int, ...]],
    ):
        self.env = env
        self._action_map = action_map

    @property
    def action_map(self):
        return self._action_map

    def probability(self, state: int, action: int) -> float:
        if self.env.is_terminal(state):
            return 0.0

        best_actions = self._action_map[state]

        if action not in best_actions:
            return 0.0

        return 1.0 / len(best_actions)

    def actions(self, state: int) -> tuple[int, ...]:
        if self.env.is_terminal(state):
            return ()

        return self._action_map[state]

    def get_best_action(self, state) -> int:
        return random.choice(self.actions(state))

    def __repr__(self):
        return f"GreedyPolicy(action_map={self._action_map})"


""" 
q(s,a) = p_h[r_H+gamma*V(s+a)] + (1-p_h)[r_T+gamma*V(s-a)]
"""


def action_value(
    env: GamblingGame,
    values: dict[int, float],
    state: int,
    action: int,
    gamma: float = 1.0,
) -> float:

    head_state = state + action
    tail_state = state - action

    head_reward = 1.0 if head_state == env.goal else 0.0
    tail_reward = 0.0

    return env.ph * (head_reward + gamma * values[head_state]) + (1.0 - env.ph) * (
        tail_reward + gamma * values[tail_state]
    )


def policy_evaluation(
    env: GamblingGame,
    policy: GreedyPolicy,
    gamma: float = 1.0,
    theta: float = 1e-6,
) -> dict[int, float]:

    values = {state: 0.0 for state in range(env.goal + 1)}

    while True:
        delta = 0.0
        new_values = values.copy()

        for state in range(1, env.goal):

            new_value = 0.0

            for action in env.get_actions(state):

                pi = policy.probability(
                    state,
                    action,
                )

                q = action_value(
                    env,
                    values,
                    state,
                    action,
                    gamma,
                )

                new_value += pi * q

            delta = max(
                delta,
                abs(new_value - values[state]),
            )

            new_values[state] = new_value

        values = new_values

        if delta < theta:
            break

    return values


def policy_improvement(
    env: GamblingGame,
    values: dict[int, float],
    state: int,
    gamma: float = 1.0,
) -> tuple[int, ...]:

    if env.is_terminal(state):
        return ()

    action_values = {
        action: action_value(
            env,
            values,
            state,
            action,
            gamma,
        )
        for action in env.get_actions(state)
    }

    best_value = max(action_values.values())

    return tuple(
        action
        for action, value in action_values.items()
        if isclose(
            value,
            best_value,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    )


def policy_iteration(
    env: GamblingGame,
    gamma: float = 1.0,
    theta: float = 1e-6,
) -> tuple[GreedyPolicy, dict[int, float]]:

    # Initial policy:
    # every legal action has equal probability
    action_map = {state: tuple(env.get_actions(state)) for state in range(1, env.goal)}

    policy = GreedyPolicy(
        env=env,
        action_map=action_map,
    )

    while True:

        # 1. Policy Evaluation
        values = policy_evaluation(
            env=env,
            policy=policy,
            gamma=gamma,
            theta=theta,
        )

        policy_stable = True
        new_action_map = {}

        # 2. Policy Improvement
        for state in range(1, env.goal):

            old_actions = policy.actions(state)

            new_actions = policy_improvement(
                env=env,
                values=values,
                state=state,
                gamma=gamma,
            )

            new_action_map[state] = new_actions

            if old_actions != new_actions:
                policy_stable = False

        policy = GreedyPolicy(
            env=env,
            action_map=new_action_map,
        )

        if policy_stable:
            return policy, values
