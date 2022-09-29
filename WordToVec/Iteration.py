from Corpus.Corpus import Corpus
from Corpus.Sentence import Sentence

from WordToVec.WordToVecParameter import WordToVecParameter


class Iteration:

    __word_count: int
    __last_word_count: int
    __word_count_actual: int
    __iteration_count: int
    __sentence_position: int
    __sentence_index: int
    __starting_alpha: float
    __alpha: float
    __corpus: Corpus
    __word_to_vec_parameter: WordToVecParameter

    def __init__(self,
                 corpus: Corpus,
                 wordToVecParameter: WordToVecParameter):
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
        self.__corpus = corpus
        self.__word_to_vec_parameter = wordToVecParameter
        self.__word_count = 0
        self.__last_word_count = 0
        self.__word_count_actual = 0
        self.__iteration_count = 0
        self.__sentence_position = 0
        self.__sentence_index = 0
        self.__starting_alpha = wordToVecParameter.getAlpha()
        self.__alpha = wordToVecParameter.getAlpha()

    def getAlpha(self) -> float:
        """
        Accessor for the alpha attribute.

        RETURNS
        -------
        float
            Alpha attribute.
        """
        return self.__alpha

    def getIterationCount(self) -> int:
        """
        Accessor for the iterationCount attribute.

        RETURNS
        -------
        int
            IterationCount attribute.
        """
        return self.__iteration_count

    def getSentenceIndex(self) -> int:
        """
        Accessor for the sentenceIndex attribute.

        RETURNS
        -------
        int
            SentenceIndex attribute
        """
        return self.__sentence_index

    def getSentencePosition(self) -> int:
        """
        Accessor for the sentencePosition attribute.

        RETURNS
        -------
        int
            SentencePosition attribute
        """
        return self.__sentence_position

    def alphaUpdate(self):
        """
        Updates the alpha parameter after 10000 words has been processed.
        """
        if self.__word_count - self.__last_word_count > 10000:
            self.__word_count_actual += self.__word_count - self.__last_word_count
            self.__last_word_count = self.__word_count
            self.__alpha = self.__starting_alpha * (1 - self.__word_count_actual /
                                                    (self.__word_to_vec_parameter.getNumberOfIterations() *
                                                     self.__corpus.numberOfWords() + 1.0))
            if self.__alpha < self.__starting_alpha * 0.0001:
                self.__alpha = self.__starting_alpha * 0.0001

    def sentenceUpdate(self, currentSentence: Sentence) -> Sentence:
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
        self.__sentence_position = self.__sentence_position + 1
        if self.__sentence_position >= currentSentence.wordCount():
            self.__word_count += currentSentence.wordCount()
            self.__sentence_index = self.__sentence_index + 1
            self.__sentence_position = 0
            if self.__sentence_index == self.__corpus.sentenceCount():
                self.__iteration_count = self.__iteration_count + 1
                self.__word_count = 0
                self.__last_word_count = 0
                self.__sentence_index = 0
                self.__corpus.shuffleSentences(1)
            return self.__corpus.getSentence(self.__sentence_index)
        return currentSentence
