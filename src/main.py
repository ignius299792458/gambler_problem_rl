import random

from gambler_problem.env import GamblingGame

state = 40
env = GamblingGame(ph=0.4, goal=100, start_capital=state)


def run_episode(policy=None, start_state: int = 40):
    t = 0
    state = start_state
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


run_episode()
