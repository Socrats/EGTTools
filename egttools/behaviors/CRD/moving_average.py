from . import AbstractCRDStrategy


class MovingAverageCRDStrategy(AbstractCRDStrategy):
    def __init__(self, a0: int, aa: int, am: int, ab: int, group_size: int):
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
