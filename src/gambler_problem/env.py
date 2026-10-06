import random


class GamblingGame:

    def __init__(self, ph: float = 0.4, goal: int = 100, start_capital=50):
        self.ph = ph
        self.goal = goal
        self.start_capital = start_capital

        self.state = self.start_capital

    def reset(self):
        self.state = self.start_capital
        return self.state

    def get_actions(self, state: int | None = None) -> list[int]:
        if state is None:
            state = self.state

        if state == 0 or state == self.goal:
            return []

        max_stake = min(state, self.goal - state)
        return list(range(1, max_stake + 1))

    def is_terminal(self, state) -> bool:
        return True if self.state in [0, 100] else False

    def step(self, action):

        if action not in self.get_actions():
            raise ValueError(f"Invalid stake {action} for capital {self.state}")

        # Coin flips
        heads = random.random() < self.ph
        if heads:
            self.state = self.state + action
        else:
            self.state = self.state - action

        # reward
        reward = 1 if self.state == self.goal else 0

        # termination
        done = self.is_terminal(self.state)

        return self.state, reward, done

    def __repr__(self):
        return f"GambingGame: {self.ph=}, {self.goal=}, {self.start_capital=}, {self.state}"


# check
if __name__ == "__main__":
    env = GamblingGame()

    print(env)
