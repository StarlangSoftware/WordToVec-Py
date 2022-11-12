import unittest

from Corpus.Corpus import Corpus
from Corpus.CorpusStream import CorpusStream
from Dictionary.VectorizedDictionary import VectorizedDictionary

from WordToVec.NeuralNetwork import NeuralNetwork
from WordToVec.WordToVecParameter import WordToVecParameter


class NeuralNetworkTest(unittest.TestCase):

    turkish: CorpusStream
    english: CorpusStream

    def setUp(self) -> None:
        self.english = CorpusStream("../english-similarity-dataset.txt")
        self.turkish = CorpusStream("../turkish-similarity-dataset.txt")

    def train(self, corpus: CorpusStream, cBow: bool) -> VectorizedDictionary:
        parameter = WordToVecParameter()
        parameter.setCbow(cBow)
        neuralNetwork = NeuralNetwork(corpus, parameter)
        return neuralNetwork.train()

    def test_TrainEnglishCBow(self):
        dictionary = self.train(self.english, True)

    def test_TrainEnglishSkipGram(self):
        dictionary = self.train(self.english, False)

    def test_TrainTurkishCBow(self):
        dictionary = self.train(self.turkish, True)

    def test_TrainTurkishSkipGram(self):
        dictionary = self.train(self.turkish, False)


if __name__ == '__main__':
    unittest.main()
