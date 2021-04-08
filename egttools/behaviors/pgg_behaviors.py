class PGGOneShotStrategy:
    def __init__(self, action):
        self.action = action

    def get_action(self):
        return self.action


def player_factory(actions):
    return [PGGOneShotStrategy(action) for action in actions]
