import unittest

from WordToVec.SemanticDataSet import SemanticDataSet


class SemanticDataSetTest(unittest.TestCase):

    def test_Spearman(self):
        semanticDataSet = SemanticDataSet("../AnlamverRel.txt")
        self.assertAlmostEqual(1.0, semanticDataSet.spearmanCorrelation(semanticDataSet), delta=0.0001)
        semanticDataSet = SemanticDataSet("../MC.txt")
        self.assertAlmostEqual(1.0, semanticDataSet.spearmanCorrelation(semanticDataSet), delta=0.0001)
        semanticDataSet = SemanticDataSet("../MEN.txt")
        self.assertAlmostEqual(1.0, semanticDataSet.spearmanCorrelation(semanticDataSet), delta=0.0001)
        semanticDataSet = SemanticDataSet("../MTurk771.txt")
        self.assertAlmostEqual(1.0, semanticDataSet.spearmanCorrelation(semanticDataSet), delta=0.0001)
        semanticDataSet = SemanticDataSet("../RareWords.txt")
        self.assertAlmostEqual(1.0, semanticDataSet.spearmanCorrelation(semanticDataSet), delta=0.0001)
        semanticDataSet = SemanticDataSet("../RG.txt")
        self.assertAlmostEqual(1.0, semanticDataSet.spearmanCorrelation(semanticDataSet), delta=0.0001)
        semanticDataSet = SemanticDataSet("../WS353.txt")
        self.assertAlmostEqual(1.0, semanticDataSet.spearmanCorrelation(semanticDataSet), delta=0.0001)
