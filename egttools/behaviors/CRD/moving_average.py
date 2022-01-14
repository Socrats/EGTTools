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

from . import AbstractCRDStrategy


class MovingAverageCRDStrategy(AbstractCRDStrategy):
    def __init__(self, a0: int, aa: int, am: int, ab: int, group_size: int):
        """
        A CRD strategy which adapts in function of a moving average of the contributions of the rest of the group.

        Parameters
        ----------
        a0: int
            how much to contribute in the initial round.
        aa: int
            how much to contribute when you have contributed above the average of the group in the previous round.
        am: int
            how much to contribute when you have contributed equal to the average of the group in the previous round.
        ab: int
            how much to contribute when you have contributed below the average of the group in the previous round.
        group_size: int
            Size of the group.
        """
        super().__init__()
        self.a0_ = a0
        self.aa_ = aa
        self.am_ = am
        self.ab_ = ab
        self.last_action_ = self.a0_
        self.nb_opponents_ = group_size - 1

    def get_action(self, time_step: int, group_contributions_prev: int):
        avg = group_contributions_prev / self.nb_opponents_
        if time_step == 0:
            return self.a0_
        else:
            if avg < self.last_action_:
                self.last_action_ = self.ab_
                return self.ab_
            elif avg > self.last_action_:
                self.last_action_ = self.aa_
                return self.aa_
            else:
                self.last_action_ = self.am_
                return self.am_

    def type(self):
        return "MovingAverageCRDStrategy({},{},{},{})".format(self.a0_, self.aa_, self.am_, self.ab_)

    def __str__(self):
        return self.type()
