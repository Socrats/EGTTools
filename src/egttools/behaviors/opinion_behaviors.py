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

from typing import List, Tuple


class Opinion:
    def __init__(self, opinion: float, tag: int):
        self.opinion = opinion
        self.tag = tag

    def get_action(self):
        pass

    @property
    def type(self):
        return "Opinion {}".format(self.opinion)


def opinion_factory(strategies: List[Tuple[float, int]]):
    return [Opinion(opinion, tag) for opinion, tag in strategies]
