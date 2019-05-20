class IdGenerator:
    __theId = 0

    @staticmethod
    def getStringId():
        IdGenerator.__theId = IdGenerator.__theId + 1
        return str(IdGenerator.__theId)