import unittest

from Corpus.Corpus import Corpus
from Dictionary.VectorizedDictionary import VectorizedDictionary

from WordToVec.NeuralNetwork import NeuralNetwork
from WordToVec.WordToVecParameter import WordToVecParameter


class NeuralNetworkTest(unittest.TestCase):

    turkish : Corpus
    english : Corpus

    def setUp(self) -> None:
        self.english = Corpus("../english-similarity-dataset.txt")
        self.turkish = Corpus("../turkish-similarity-dataset.txt")

    def train(self, corpus: Corpus, cBow: bool) -> VectorizedDictionary:
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
