import numpy as np
from .abstract_cpr_strategy import AbstractCPRStrategy


def high_extraction(a: float, b: float, group_size: int) -> float:
    return a / (b * group_size)


def fair_extraction(a: float, b: float, group_size: int) -> float:
    group_optimal = a / (2 * b)
    return group_optimal / group_size


def nash_extraction(a: float, b: float, group_size: int) -> float:
    group_max = (group_size / (group_size + 1)) * (a / b)
    return group_max / group_size


def payoff_no_commitment(a: float, b: float, extraction: float, group_extraction: float) -> float:
    return (extraction / group_extraction) * ((a * group_extraction) - (b * np.power(group_extraction, 2)))


class FixedExtraction(AbstractCPRStrategy):

    def __init__(self, extraction: float, accepts_commitment: bool):
        self.extraction = extraction
        self.accepts_commitment = accepts_commitment

    def get_extraction(self, a: float, b: float, group_size: int, commitment: bool = False) -> float:
        return self.extraction

    @staticmethod
    def get_payoff(a: float, b: float, extraction: float, group_extraction: float, fine: float = 0,
                   cost: float = 0, commitment: bool = False) -> float:
        return payoff_no_commitment(a, b, extraction, group_extraction)

    def would_like_to_commit(self) -> bool:
        return self.accepts_commitment

    def proposes_commitment(self) -> bool:
        return False

    def is_commitment_validated(self, nb_committers: int) -> bool:
        return False

    def type(self) -> str:
        return f"EXT{self.extraction}{self.accepts_commitment}"

    def __str__(self) -> str:
        return f"FixedExtraction_e{self.extraction}_c{self.accepts_commitment}"


class FairExtraction(AbstractCPRStrategy):

    def get_extraction(self, a: float, b: float, group_size: int, commitment: bool = False) -> float:
        return fair_extraction(a, b, group_size)

    @staticmethod
    def get_payoff(a: float, b: float, extraction: float, group_extraction: float, fine: float = 0,
                   cost: float = 0, commitment: bool = False) -> float:
        return payoff_no_commitment(a, b, extraction, group_extraction)

    def would_like_to_commit(self) -> bool:
        return True

    def proposes_commitment(self) -> bool:
        return False

    def is_commitment_validated(self, nb_committers: int) -> bool:
        if nb_committers > 0:
            return True
        else:
            return False

    def type(self) -> str:
        return "FAIR"

    def __str__(self) -> str:
        return "FairExtraction"


class HighExtraction(AbstractCPRStrategy):

    def get_extraction(self, a: float, b: float, group_size: int, commitment: bool = False) -> float:
        return high_extraction(a, b, group_size)

    @staticmethod
    def get_payoff(a: float, b: float, extraction: float, group_extraction: float, fine: float = 0,
                   cost: float = 0, commitment: bool = False) -> float:
        return payoff_no_commitment(a, b, extraction, group_extraction)

    def would_like_to_commit(self) -> bool:
        return False

    def proposes_commitment(self) -> bool:
        return False

    def is_commitment_validated(self, nb_committers: int) -> bool:
        return False

    def type(self) -> str:
        return "HIGH"

    def __str__(self) -> str:
        return "HighExtraction"


class NashExtraction(AbstractCPRStrategy):

    def get_extraction(self, a: float, b: float, group_size: int, commitment: bool = False) -> float:
        return nash_extraction(a, b, group_size)

    @staticmethod
    def get_payoff(a: float, b: float, extraction: float, group_extraction: float, fine: float = 0,
                   cost: float = 0, commitment: bool = False) -> float:
        return payoff_no_commitment(a, b, extraction, group_extraction)

    def would_like_to_commit(self) -> bool:
        return False

    def proposes_commitment(self) -> bool:
        return False

    def is_commitment_validated(self, nb_committers: int) -> bool:
        return False

    def type(self) -> str:
        return "NASH"

    def __str__(self) -> str:
        return "NashExtraction"


class CommitmentStrategy(AbstractCPRStrategy):
    def __init__(self, commitment_threshold):
        self.commitment_threshold = commitment_threshold

    def get_extraction(self, a: float, b: float, group_size: int, commitment: bool = False) -> float:
        if commitment:  # extracts Fair
            return fair_extraction(a, b, group_size)
        else:  # extracts High
            return high_extraction(a, b, group_size)

    @staticmethod
    def get_payoff(a: float, b: float, extraction: float, group_extraction: float, fine: float = 0,
                   cost: float = 0, commitment: bool = False) -> float:
        if commitment:
            return payoff_no_commitment(a, b, extraction, group_extraction) - cost
        else:
            return payoff_no_commitment(a, b, extraction, group_extraction)

    def would_like_to_commit(self) -> bool:
        return True

    def proposes_commitment(self) -> bool:
        return True

    def is_commitment_validated(self, nb_committers: int) -> bool:
        return False if nb_committers < self.commitment_threshold else True

    def type(self) -> str:
        return f"COM{self.commitment_threshold}"

    def __str__(self) -> str:
        return f"CommitmentStrategy{self.commitment_threshold}"


class FakeStrategy(AbstractCPRStrategy):

    def get_extraction(self, a: float, b: float, group_size: int, commitment: bool = False) -> float:
        return high_extraction(a, b, group_size)

    @staticmethod
    def get_payoff(a: float, b: float, extraction: float, group_extraction: float, fine: float = 0,
                   cost: float = 0, commitment: bool = False) -> float:
        if commitment:
            return payoff_no_commitment(a, b, extraction, group_extraction) - fine
        else:
            return payoff_no_commitment(a, b, extraction, group_extraction)

    def would_like_to_commit(self) -> bool:
        return True

    def proposes_commitment(self) -> bool:
        return False

    def is_commitment_validated(self, nb_committers: int) -> bool:
        return True if nb_committers > 0 else False

    def type(self) -> str:
        return "FAKE"

    def __str__(self) -> str:
        return "FakeStrategy"


class FreeStrategy(AbstractCPRStrategy):

    def get_extraction(self, a: float, b: float, group_size: int, commitment: bool = False) -> float:
        if commitment:
            return fair_extraction(a, b, group_size)
        else:
            return high_extraction(a, b, group_size)

    @staticmethod
    def get_payoff(a: float, b: float, extraction: float, group_extraction: float, fine: float = 0,
                   cost: float = 0, commitment: bool = False) -> float:
        return payoff_no_commitment(a, b, extraction, group_extraction)

    def would_like_to_commit(self) -> bool:
        return True

    def proposes_commitment(self) -> bool:
        return False

    def is_commitment_validated(self, nb_committers: int) -> bool:
        return True if nb_committers > 0 else False

    def type(self) -> str:
        return "FREE"

    def __str__(self) -> str:
        return "FreeStrategy"
