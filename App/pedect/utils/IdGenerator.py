class IdGenerator:
    theId = 0

    @staticmethod
    def getStringId():
        IdGenerator.theId = IdGenerator.theId + 1
        return str(IdGenerator.theId)