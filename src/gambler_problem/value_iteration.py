from gambler_problem.env import GamblingGame
from gambler_problem.policy import GreedyPolicy, action_value, policy_improvement


def value_iteration(
    env: GamblingGame,
    gamma: float = 1.0,
    theta: float = 1e-6,
) -> tuple[GreedyPolicy, dict[int, float]]:

    values = {state: 0.0 for state in range(env.goal + 1)}

    while True:

        delta = 0.0
        new_values = values.copy()

        for state in range(1, env.goal):

            best_value = max(
                action_value(
                    env=env,
                    values=values,
                    state=state,
                    action=action,
                    gamma=gamma,
                )
                for action in env.get_actions(state)
            )

            delta = max(
                delta,
                abs(best_value - values[state]),
            )

            new_values[state] = best_value

        values = new_values

        if delta < theta:
            break

    # Extract optimal policy
    action_map = {
        state: policy_improvement(
            env=env,
            values=values,
            state=state,
            gamma=gamma,
        )
        for state in range(1, env.goal)
    }

    policy = GreedyPolicy(
        env=env,
        action_map=action_map,
    )

    return policy, values
