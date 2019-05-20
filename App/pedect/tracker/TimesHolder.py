class TimesHolder:
    time0, time1, time2, time3, time4 = 0.0, 0.0, 0.0, 0.0, 0.0
    trackerAccessed = 0
    cachedTrackerAccessed = 0

    @staticmethod
    def getTimes():
        return TimesHolder.time0, TimesHolder.time1, TimesHolder.time2, TimesHolder.time3, TimesHolder.time4
