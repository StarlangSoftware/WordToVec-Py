class WordPair:

    word1: str
    word2: str
    related_by: float

    def __init__(self,
                 word1: str,
                 word2: str,
                 relatedBy: float):
        """
        Constructor of the WordPair object. WordPair stores the information about two words and their similarity scores.
        :param word1: First word
        :param word2: Second word
        :param relatedBy: Similarity score between first and second word.
        """
        self.word1 = word1
        self.word2 = word2
        self.related_by = relatedBy

    def __eq__(self, other):
        return self.word1 == other.word1 and self.word2 == other.word2

    def getRelatedBy(self) -> float:
        """
        Accessor for the similarity score.
        :return: Similarity score.
        """
        return self.related_by

    def setRelatedBy(self, relatedBy: float):
        """
        Mutator for the similarity score.
        :param relatedBy: New similarity score
        """
        self.related_by = relatedBy

    def getWord1(self) -> str:
        """
        Accessor for the first word.
        :return: First word.
        """
        return self.word1

    def getWord2(self) -> str:
        """
        Accessor for the second word.
        :return: Second word.
        """
        return self.word2

    def __repr__(self):
        return f"{self.word1} {self.word2}"
