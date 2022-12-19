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


class GoalBasedCRDStrategy(AbstractCRDStrategy):
    def __init__(self, a0: int, s1r0: int, s1r1: int, s1r2: int, s1r3: int, s1r4: int,
                 s2r0: int, s2r1: int, s2r2: int, s2r3: int, s2r4: int, switch: int, group_size: int):
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
        self.s1_ = (s1r0, s1r1, s1r2, s1r3, s1r4)
        self.s2_ = (s2r0, s2r1, s2r2, s2r3, s2r4)
        self.a0_ = a0
        self.switch_ = switch
        self.last_action_ = self.a0_
        self.nb_opponents_ = group_size - 1
        self.public_account_ = 0

    def get_action(self, time_step: int, group_contributions_prev: int):
        avg = round(group_contributions_prev / self.nb_opponents_)

        if time_step == 0:
            self.public_account_ += self.a0_
            return self.a0_
        else:
            self.public_account_ += group_contributions_prev

            if self.public_account_ < self.switch_:  # S1
                self.last_action_ = self.s1_[avg]
            else:  # S2
                self.last_action_ = self.s2_[avg]

            # Update public account with the current action and return it
            self.public_account_ += self.last_action_
            return self.last_action_

    def type(self):
        return "GoalBasedCRDStrategy({},{},{},{},{},{},{},{},{},{},{},{})".format(self.a0_, *self.s1_, self.switch_,
                                                                                  *self.s2_)

    def __str__(self):
        return self.type()
