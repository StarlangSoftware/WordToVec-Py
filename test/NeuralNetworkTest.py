import unittest

from Corpus.AbstractCorpus import AbstractCorpus
from Corpus.Corpus import Corpus
from Dictionary.VectorizedDictionary import VectorizedDictionary

from WordToVec.NeuralNetwork import NeuralNetwork
from WordToVec.SemanticDataSet import SemanticDataSet
from WordToVec.WordToVecParameter import WordToVecParameter


class NeuralNetworkTest(unittest.TestCase):

    turkish: AbstractCorpus
    english: AbstractCorpus
    mc: SemanticDataSet
    rg: SemanticDataSet
    ws: SemanticDataSet
    men: SemanticDataSet
    mturk: SemanticDataSet
    rare: SemanticDataSet
    av: SemanticDataSet

    def setUp(self) -> None:
        self.english = Corpus("../english-xs.txt")
        self.turkish = Corpus("../turkish-xs.txt")
        self.mc = SemanticDataSet("../MC.txt")
        self.rg = SemanticDataSet("../RG.txt")
        self.ws = SemanticDataSet("../WS353.txt")
        self.men = SemanticDataSet("../MEN.txt")
        self.mturk = SemanticDataSet("../MTurk771.txt")
        self.rare = SemanticDataSet("../RareWords.txt")
        self.av = SemanticDataSet("../AnlamverRel.txt")

    def train(self, corpus: AbstractCorpus, cBow: bool) -> VectorizedDictionary:
        parameter = WordToVecParameter()
        parameter.setCbow(cBow)
        neuralNetwork = NeuralNetwork(corpus, parameter)
        print(neuralNetwork.vocabularySize())
        return neuralNetwork.train()

    def test_TrainEnglishCBow(self):
        dictionary = self.train(self.english, True)
        mc2 = self.mc.calculateSimilarities(dictionary)
        print("(" + str(self.mc.size()) + ") " + str(self.mc.spearmanCorrelation(mc2)))
        rg2 = self.rg.calculateSimilarities(dictionary)
        print("(" + str(self.rg.size()) + ") " + str(self.rg.spearmanCorrelation(rg2)))
        ws2 = self.ws.calculateSimilarities(dictionary)
        print("(" + str(self.ws.size()) + ") " + str(self.ws.spearmanCorrelation(ws2)))
        men2 = self.men.calculateSimilarities(dictionary)
        print("(" + str(self.men.size()) + ") " + str(self.men.spearmanCorrelation(men2)))
        mturk2 = self.mturk.calculateSimilarities(dictionary)
        print("(" + str(self.mturk.size()) + ") " + str(self.mturk.spearmanCorrelation(mturk2)))
        rare2 = self.rare.calculateSimilarities(dictionary)
        print("(" + str(self.rare.size()) + ") " + str(self.rare.spearmanCorrelation(rare2)))

    def test_wordVectors(self):
        dictionary = VectorizedDictionary(None, "../vectors-english-xs.txt")
        mc2 = self.mc.calculateSimilarities(dictionary)
        print("(" + str(self.mc.size()) + ") " + str(self.mc.spearmanCorrelation(mc2)))
        rg2 = self.rg.calculateSimilarities(dictionary)
        print("(" + str(self.rg.size()) + ") " + str(self.rg.spearmanCorrelation(rg2)))
        ws2 = self.ws.calculateSimilarities(dictionary)
        print("(" + str(self.ws.size()) + ") " + str(self.ws.spearmanCorrelation(ws2)))
        men2 = self.men.calculateSimilarities(dictionary)
        print("(" + str(self.men.size()) + ") " + str(self.men.spearmanCorrelation(men2)))
        mturk2 = self.mturk.calculateSimilarities(dictionary)
        print("(" + str(self.mturk.size()) + ") " + str(self.mturk.spearmanCorrelation(mturk2)))
        rare2 = self.rare.calculateSimilarities(dictionary)
        print("(" + str(self.rare.size()) + ") " + str(self.rare.spearmanCorrelation(rare2)))

    def test_TrainEnglishSkipGram(self):
        dictionary = self.train(self.english, False)
        mc2 = self.mc.calculateSimilarities(dictionary)
        print("(" + str(self.mc.size()) + ") " + str(self.mc.spearmanCorrelation(mc2)))
        rg2 = self.rg.calculateSimilarities(dictionary)
        print("(" + str(self.rg.size()) + ") " + str(self.rg.spearmanCorrelation(rg2)))
        ws2 = self.ws.calculateSimilarities(dictionary)
        print("(" + str(self.ws.size()) + ") " + str(self.ws.spearmanCorrelation(ws2)))
        men2 = self.men.calculateSimilarities(dictionary)
        print("(" + str(self.men.size()) + ") " + str(self.men.spearmanCorrelation(men2)))
        mturk2 = self.mturk.calculateSimilarities(dictionary)
        print("(" + str(self.mturk.size()) + ") " + str(self.mturk.spearmanCorrelation(mturk2)))
        rare2 = self.rare.calculateSimilarities(dictionary)
        print("(" + str(self.rare.size()) + ") " + str(self.rare.spearmanCorrelation(rare2)))

    def test_TrainTurkishCBow(self):
        dictionary = self.train(self.turkish, True)
        av2 = self.av.calculateSimilarities(dictionary)
        print("(" + str(self.av.size()) + ") " + str(self.av.spearmanCorrelation(av2)))

    def test_TrainTurkishSkipGram(self):
        dictionary = self.train(self.turkish, False)
        av2 = self.av.calculateSimilarities(dictionary)
        print("(" + str(self.av.size()) + ") " + str(self.av.spearmanCorrelation(av2)))


if __name__ == '__main__':
    unittest.main()
