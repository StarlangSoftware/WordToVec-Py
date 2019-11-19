class WordToVecParameter:

    __layerSize: int
    __cbow : bool
    __alpha : float
    __window : int
    __hierarchicalSoftMax : bool
    __negativeSamplingSize : int
    __numberOfIterations : int

    """
    Empty constructor for Word2Vec parameter
    """
    def __init__(self):
        self.__alpha = 0.025
        self.__cbow = True
        self.__hierarchicalSoftMax = False
        self.__layerSize = 100
        self.__negativeSamplingSize = 5
        self.__numberOfIterations = 3
        self.__window = 5

    """
    Accessor for layerSize attribute.

    RETURNS
    -------
    int
        Size of the word vectors.
    """
    def getLayerSize(self) -> int:
        return self.__layerSize

    """
    Accessor for CBow attribute.
    
    RETURNS
    -------
    bool
        True is CBow will be applied, false otherwise.
    """
    def isCbow(self) -> bool:
        return self.__cbow

    """
    Accessor for the alpha attribute.

    RETURNS
    -------
    float
        Current learning rate alpha.
    """
    def getAlpha(self) -> float:
        return self.__alpha

    """
    Accessor for the window size attribute.
    
    RETURNS
    -------
    int
        Current window size.
    """
    def getWindow(self) -> int:
        return self.__window

    """
    Accessor for the hierarchicalSoftMax attribute.

    RETURNS
    -------
    bool
        If hierarchical softmax will be applied, returns true; false otherwise.
    """
    def isHierarchicalSoftMax(self) -> bool:
        return self.__hierarchicalSoftMax

    """
    Accessor for the negativeSamplingSize attribute.
    
    RETURNS
    -------
    int
        Number of negative samples that will be withdrawn.
    """
    def getNegativeSamplingSize(self) -> int:
        return self.__negativeSamplingSize

    """
    Accessor for the numberOfIterations attribute.

    RETURNS
    -------
    int
        Number of epochs to train the network.
    """
    def getNumberOfIterations(self) -> int:
        return self.__numberOfIterations

    """
    Mutator for the layerSize attribute.

    PARAMETERS
    ----------
    layerSize : int
        New size of the word vectors.
    """
    def setLayerSize(self, layerSize: int):
        self.__layerSize = layerSize

    """
    Mutator for cBow attribute

    PARAMETERS
    ----------
    cbow : bool
        True if CBow applied; false if SkipGram applied.
    """
    def setCbow(self, cbow: bool):
        self.__cbow = cbow

    """
    Mutator for alpha attribute
    
    PARAMETERS
    ----------
    alpha : float
        New learning rate.
    """
    def setAlpha(self, alpha: float):
        self.__alpha = alpha

    """
    Mutator for the window size attribute.
    
    PARAMETERS
    ----------
    window : int
        New window size.
    """
    def setWindow(self, window: int):
        self.__window = window

    """
    Mutator for the hierarchicalSoftMax attribute.
    
    PARAMETERS
    ----------
    hierarchicalSoftMax : bool
        True is hierarchical softMax applied; false otherwise.
    """
    def setHierarchialSoftMax(self, hierarchicalSoftMax: bool):
        self.__hierarchicalSoftMax = hierarchicalSoftMax

    """
    Mutator for the negativeSamplingSize attribute.
    
    PARAMETERS
    ----------
    negativeSamplingSize : int
        New number of negative instances that will be withdrawn.
    """
    def setNegativeSamplingSize(self, negativeSamplingSize: int):
        self.__negativeSamplingSize = negativeSamplingSize

    """
    Mutator for the numberOfIterations attribute.
    
    PARAMETERS
    ----------
    numberOfIterations : int
        New number of iterations.
    """
    def setNumberOfIterations(self, numberOfIterations: int):
        self.__numberOfIterations = numberOfIterations
