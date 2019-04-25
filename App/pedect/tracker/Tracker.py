from typing import Tuple


class Tracker:

    @NotImplementedError
    def track(self, uniqueId: str, image, bbox: Tuple[int, int, int, int] = None) -> Tuple[int, int, int, int]:
        pass

    @NotImplementedError
    def clearTracker(self):
        pass