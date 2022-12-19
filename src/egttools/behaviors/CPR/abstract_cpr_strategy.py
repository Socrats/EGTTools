from abc import ABC, abstractmethod


class AbstractCPRStrategy(ABC):
    @abstractmethod
    def get_extraction(self, a: float, b: float, group_size: int, commitment: bool = False) -> float:
        pass

    @staticmethod
    @abstractmethod
    def get_payoff(a: float, b: float, extraction: float, group_extraction: float, fine: float = 0,
                   cost: float = 0, commitment: bool = False) -> float:
        pass

    @abstractmethod
    def would_like_to_commit(self) -> bool:
        pass

    @abstractmethod
    def proposes_commitment(self) -> bool:
        pass

    @abstractmethod
    def is_commitment_validated(self, nb_committers: int) -> bool:
        pass

    @abstractmethod
    def type(self) -> str:
        pass

    def __str__(self) -> str:
        return "AbstractCPRStrategy"
