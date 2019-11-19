from Dictionary.Word import Word


class VocabularyWord(Word):

    __count : int
    __code : list
    __point : list
    __codeLength : int
    MAX_CODE_LENGTH = 40

    """
    Constructor for a VocabularyWord. The constructor gets name and count values and sets the corresponding
    attributes. It also initializes the code and point arrays for this word.

    PARAMETERS
    ----------
    name : str
        Lemma of the word
    count : int
        Number of occurrences of this word in the corpus
    """
    def __init__(self, name: str, count: int):
        super().__init__(name)
        self.__count = count
        code = []
        point = []
        self.__codeLength = 0

    def __lt__(self, other) -> bool:
        return self.__count < other.__count

    def __gt__(self, other) -> bool:
        return self.__count > other.__count

    def __eq__(self, other) -> bool:
        return self.__count == other.__count

    """
    Accessor for the count attribute.
    
    RETURNS
    -------
    int
        Number of occurrences of this word.
    """
    def getCount(self) -> int:
        return self.__count

    """
    Mutator for codeLength attribute.
    
    PARAMETERS
    ----------
    codeLength : int
        New value for the codeLength.
    """
    def setCodeLength(self, codeLength: int):
        self.__codeLength = codeLength

    """
    Mutator for code attribute.

    PARAMETERS
    ----------
    index : int
        Index of the code
    value : int
        New value for that indexed element of code.
    """
    def setCode(self, index: int, value: int):
        self.__code[index] = value

    """
    Mutator for point attribute.

    PARAMETERS
    ----------
    index : int
        Index of the point
    value : int
        New value for that indexed element of point.
    """
    def setPoint(self, index: int, value: int):
        self.__point[index] = value

    """
    Accessor for the codeLength attribute.

    RETURNS
    -------
    int
        Length of the Huffman code for this word.
    """
    def getCodeLength(self) -> int:
        return self.__codeLength

    """
    Accessor for point attribute.

    PARAMETERS
    ----------
    index : int
        Index of the point.

    RETURNS
    -------
    int
        Value for that indexed element of point.
    """
    def getPoint(self, index: int) -> int:
        return self.__point[index]

    """
    Accessor for code attribute.

    PARAMETERS
    ----------
    index : int
        Index of the code.

    RETURNS
    -------
    int
        Value for that indexed element of code.
    """
    def getCode(self, index: int) -> int:
        return self.__code[index]
