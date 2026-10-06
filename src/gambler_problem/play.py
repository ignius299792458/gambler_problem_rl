import random

from gambler_problem.env import GamblingGame
from gambler_problem.policy import GreedyPolicy, policy_iteration


def run_episode(env: GamblingGame, policy: GreedyPolicy | None = None, state: int = 50):
    t = 0
    while True:
        action = (
            random.choice(env.get_actions())
            if policy is None
            else policy.get_best_action(state)
        )

        next_state, reward, done = env.step(action)

        print(
            f"S_{t}:{state}, A_{t}:{action} -> R_{t+1}:{reward}, S_{t+1}:{next_state}, {done}"
        )

        state = next_state

        if done:
            print(f"Episode end at time-step t={t}")
            break

        t += 1


# test
if __name__ == "__main__":
    env = GamblingGame()

    # run episode with random policy
    run_episode(env)

    # run episode with policy iterated
    from gambler_problem.policy import policy_iteration

    greedy_policy, vpi_values = policy_iteration(env)

    run_episode(env, greedy_policy)

    # run with value iteration
