# Copyright (c) 2019-2021  Elias Fernandez
#
# This file is part of EGTtools.
#
# EGTtools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EGTtools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EGTtools.  If not, see <http://www.gnu.org/licenses/>

"""
The `behaviors.pgg_behaviors` submodule contains strategies which can
be used with `egttools.games.pgg` game.
"""

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
