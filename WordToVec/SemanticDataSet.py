from __future__ import annotations

from functools import cmp_to_key

from Dictionary.VectorizedDictionary import VectorizedDictionary
from Dictionary.VectorizedWord import VectorizedWord

from WordToVec.WordPair import WordPair


class SemanticDataSet:

    __pairs: list[WordPair]

    def __init__(self, file_name: str = None):
        self.__pairs = []
        if file_name is not None:
            file = open(file_name, "r")
            lines = file.readlines()
            file.close()
            for line in lines:
                items = line.split(" ")
                self.__pairs.append(WordPair(items[0], items[1], float(items[2])))

    def calculateSimilarities(self, dictionary: VectorizedDictionary) -> SemanticDataSet:
        result = SemanticDataSet()
        i = 0
        while i < len(self.__pairs):
            word1 = self.__pairs[i].getWord1()
            word2 = self.__pairs[i].getWord2()
            vectorized_word1 = dictionary.getWord(word1)
            vectorized_word2 = dictionary.getWord(word2)
            if word1 is not None and word2 is not None and \
                    isinstance(vectorized_word1, VectorizedWord) and isinstance(vectorized_word2, VectorizedWord):
                similarity = vectorized_word1.getVector().cosineSimilarity(vectorized_word2.getVector())
                result.__pairs.append(WordPair(word1, word2, similarity))
            else:
                self.__pairs.pop(i)
                i = i - 1
            i = i + 1
        return result

    def size(self) -> int:
        return len(self.__pairs)

    @staticmethod
    def wordPairComparator(wordPairA: WordPair, wordPairB: WordPair):
        if wordPairA.getRelatedBy() > wordPairB.getRelatedBy():
            return -1
        elif wordPairA.getRelatedBy() < wordPairB.getRelatedBy():
            return 1
        else:
            return 0

    def sort(self):
        self.__pairs.sort(key=cmp_to_key(self.wordPairComparator))

    def index(self, wordPair: WordPair):
        for i in range(len(self.__pairs)):
            if wordPair == self.__pairs[i]:
                return i
        return -1

    def spearmanCorrelation(self, semanticDataSet: SemanticDataSet) -> float:
        total = 0
        self.sort()
        semanticDataSet.sort()
        for i in range(len(self.__pairs)):
            rank1 = i + 1
            if semanticDataSet.index(self.__pairs[i]) != -1:
                rank2 = semanticDataSet.index(self.__pairs[i]) + 1
            else:
                return -1
            di = rank1 - rank2
            total = total + 6 * di * di
        n = len(self.__pairs)
        ratio = total / (n * (n * n - 1))
        return 1 - ratio
