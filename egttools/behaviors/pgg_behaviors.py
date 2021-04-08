from typing import List


class PGGOneShotStrategy:
    def __init__(self, action: int) -> None:
        self.action = action

    def get_action(self) -> int:
        return self.action

    @property
    def type(self) -> str:
        return "PGGOneShotStrategy"


def player_factory(actions: List[int]) -> List[PGGOneShotStrategy]:
    return [PGGOneShotStrategy(action) for action in actions]
