from abc import abstractmethod
from typing import Tuple, Sequence


class Tracker:

    def __init__(self):
        self.trackerType = ""

    @abstractmethod
    def track(self, uniqueId: str, image, bbox: Tuple[int, int, int, int] = None, imageHash: int = None) -> \
            Tuple[int, int, int, int]:
        pass

    # def clearTracker(self):
    #     pass

    @abstractmethod
    def parallelizable(self) -> bool:
        pass

    def trackAll(self, uniqueIds: Sequence[str], image, imageHash: int = None) -> dict:
        rez = {}
        for uniqueId in uniqueIds:
            rez[uniqueId] = self.track(uniqueId, image, None, imageHash)
        return rez