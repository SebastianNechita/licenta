from abc import abstractmethod
from typing import Tuple, Sequence


class Tracker:

    @abstractmethod
    def track(self, uniqueId: str, image, bbox: Tuple[int, int, int, int] = None) -> Tuple[int, int, int, int]:
        pass

    def clearTracker(self):
        pass

    @abstractmethod
    def parallelizable(self) -> bool:
        pass

    def trackAll(self, uniqueIds: Sequence[str], image) -> dict:
        rez = {}
        for uniqueId in uniqueIds:
            rez[uniqueId] = self.track(uniqueId, image)
        return rez