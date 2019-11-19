from Corpus.Corpus import Corpus
from Corpus.Sentence import Sentence

from WordToVec.WordToVecParameter import WordToVecParameter


class Iteration:

    __wordCount: int
    __lastWordCount: int
    __wordCountActual: int
    __iterationCount: int
    __sentencePosition: int
    __sentenceIndex: int
    __startingAlpha: float
    __alpha: float
    __corpus: Corpus
    __wordToVecParameter: WordToVecParameter

    """
    Constructor for the Iteration class. Get corpus and parameter as input, sets the corresponding
    parameters.

    PARAMETERS
    ----------
    corpus : Corpus
        Corpus used to train word vectors using Word2Vec algorithm.
    wordToVecParameter : WordToVecParameter
        Parameters of the Word2Vec algorithm.
    """
    def __init__(self, corpus: Corpus, wordToVecParameter: WordToVecParameter):
        self.__corpus = corpus
        self.__wordToVecParameter = wordToVecParameter
        self.__wordCount = 0
        self.__lastWordCount = 0
        self.__wordCountActual = 0
        self.__iterationCount = 0
        self.__sentencePosition = 0
        self.__sentenceIndex = 0
        self.__startingAlpha = wordToVecParameter.getAlpha()
        self.__alpha = wordToVecParameter.getAlpha()

    """
    Accessor for the alpha attribute.

    RETURNS
    -------
    float
        Alpha attribute.
    """
    def getAlpha(self) -> float:
        return self.__alpha

    """
    Accessor for the iterationCount attribute.
    
    RETURNS
    -------
    int
        IterationCount attribute.
    """
    def getIterationCount(self) -> int:
        return self.__iterationCount

    """
    Accessor for the sentenceIndex attribute.
    
    RETURNS
    -------
    int
        SentenceIndex attribute
    """
    def getSentenceIndex(self) -> int:
        return self.__sentenceIndex

    """
    Accessor for the sentencePosition attribute.

    RETURNS
    -------
    int
        SentencePosition attribute
    """
    def getSentencePosition(self) -> int:
        return self.__sentencePosition

    """
    Updates the alpha parameter after 10000 words has been processed.
    """
    def alphaUpdate(self):
        if self.__wordCount - self.__lastWordCount > 10000:
            self.__wordCountActual += self.__wordCount - self.__lastWordCount
            self.__lastWordCount = self.__wordCount
            self.__alpha = self.__startingAlpha * (1 - self.__wordCountActual / (self.__wordToVecParameter.getNumberOfIterations() * self.__corpus.numberOfWords() + 1.0))
            if self.__alpha < self.__startingAlpha * 0.0001:
                self.__alpha = self.__startingAlpha * 0.0001

    """
    Updates sentencePosition, sentenceIndex (if needed) and returns the current sentence processed. If one sentence
    is finished, the position shows the beginning of the next sentence and sentenceIndex is incremented. If the
    current sentence is the last sentence, the system shuffles the sentences and returns the first sentence.

    PARAMETERS
    ----------
    currentSentence : Sentence
        Current sentence processed.
        
    RETURNS
    -------
    Sentence
        If current sentence is not changed, currentSentence; if changed the next sentence; if next sentence is 
        the last sentence; shuffles the corpus and returns the first sentence.
    """
    def sentenceUpdate(self, currentSentence: Sentence) -> Sentence:
        self.__sentencePosition = self.__sentencePosition + 1
        if self.__sentencePosition >= currentSentence.wordCount():
            self.__wordCount += currentSentence.wordCount()
            self.__sentenceIndex = self.__sentenceIndex + 1
            self.__sentencePosition = 0
            if self.__sentenceIndex == self.__corpus.sentenceCount():
                self.__iterationCount = self.__iterationCount + 1
                self.__wordCount = 0
                self.__lastWordCount = 0
                self.__sentenceIndex = 0
                self.__corpus.shuffleSentences(1)
            return self.__corpus.getSentence(self.__sentenceIndex)
        return currentSentence
