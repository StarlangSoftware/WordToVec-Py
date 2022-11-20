class WordPair:

    word1: str
    word2: str
    related_by: float

    def __init__(self,
                 word1: str,
                 word2: str,
                 relatedBy: float):
        self.word1 = word1
        self.word2 = word2
        self.related_by = relatedBy

    def __eq__(self, other):
        return self.word1 == other.word1 and self.word2 == other.word2

    def getRelatedBy(self) -> float:
        return self.related_by

    def setRelatedBy(self, relatedBy: float):
        self.related_by = relatedBy

    def getWord1(self) -> str:
        return self.word1

    def getWord2(self) -> str:
        return self.word2

    def __repr__(self):
        return f"{self.word1} {self.word2}"
