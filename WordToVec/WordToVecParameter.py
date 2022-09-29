class WordToVecParameter:

    __layer_size: int
    __cbow: bool
    __alpha: float
    __window: int
    __hierarchical_soft_max: bool
    __negative_sampling_size: int
    __number_of_iterations: int
    __seed: int

    def __init__(self):
        """
        Empty constructor for Word2Vec parameter
        """
        self.__alpha = 0.025
        self.__cbow = True
        self.__hierarchical_soft_max = False
        self.__layer_size = 100
        self.__negative_sampling_size = 5
        self.__number_of_iterations = 3
        self.__window = 5
        self.__seed = 1

    def getLayerSize(self) -> int:
        """
        Accessor for layerSize attribute.

        RETURNS
        -------
        int
            Size of the word vectors.
        """
        return self.__layer_size

    def isCbow(self) -> bool:
        """
        Accessor for CBow attribute.

        RETURNS
        -------
        bool
            True is CBow will be applied, false otherwise.
        """
        return self.__cbow

    def getAlpha(self) -> float:
        """
        Accessor for the alpha attribute.

        RETURNS
        -------
        float
            Current learning rate alpha.
        """
        return self.__alpha

    def getWindow(self) -> int:
        """
        Accessor for the window size attribute.

        RETURNS
        -------
        int
            Current window size.
        """
        return self.__window

    def isHierarchicalSoftMax(self) -> bool:
        """
        Accessor for the hierarchicalSoftMax attribute.

        RETURNS
        -------
        bool
            If hierarchical softmax will be applied, returns true; false otherwise.
        """
        return self.__hierarchical_soft_max

    def getNegativeSamplingSize(self) -> int:
        """
        Accessor for the negativeSamplingSize attribute.

        RETURNS
        -------
        int
            Number of negative samples that will be withdrawn.
        """
        return self.__negative_sampling_size

    def getNumberOfIterations(self) -> int:
        """
        Accessor for the numberOfIterations attribute.

        RETURNS
        -------
        int
            Number of epochs to train the network.
        """
        return self.__number_of_iterations

    def getSeed(self) -> int:
        """
        Accessor for the seed attribute.

        RETURNS
        -------
        int
            Seed to train the network.
        """
        return self.__seed

    def setLayerSize(self, layerSize: int):
        """
        Mutator for the layerSize attribute.

        PARAMETERS
        ----------
        layerSize : int
            New size of the word vectors.
        """
        self.__layer_size = layerSize

    def setCbow(self, cbow: bool):
        """
        Mutator for cBow attribute

        PARAMETERS
        ----------
        cbow : bool
            True if CBow applied; false if SkipGram applied.
        """
        self.__cbow = cbow

    def setAlpha(self, alpha: float):
        """
        Mutator for alpha attribute

        PARAMETERS
        ----------
        alpha : float
            New learning rate.
        """
        self.__alpha = alpha

    def setWindow(self, window: int):
        """
        Mutator for the window size attribute.

        PARAMETERS
        ----------
        window : int
            New window size.
        """
        self.__window = window

    def setHierarchialSoftMax(self, hierarchicalSoftMax: bool):
        """
        Mutator for the hierarchicalSoftMax attribute.

        PARAMETERS
        ----------
        hierarchicalSoftMax : bool
            True is hierarchical softMax applied; false otherwise.
        """
        self.__hierarchical_soft_max = hierarchicalSoftMax

    def setNegativeSamplingSize(self, negativeSamplingSize: int):
        """
        Mutator for the negativeSamplingSize attribute.

        PARAMETERS
        ----------
        negativeSamplingSize : int
            New number of negative instances that will be withdrawn.
        """
        self.__negative_sampling_size = negativeSamplingSize

    def setNumberOfIterations(self, numberOfIterations: int):
        """
        Mutator for the numberOfIterations attribute.

        PARAMETERS
        ----------
        numberOfIterations : int
            New number of iterations.
        """
        self.__number_of_iterations = numberOfIterations

    def setSeed(self, seed: int):
        """
        Mutator for the seed attribute.

        PARAMETERS
        ----------
        seed : int
            New seed.
        """
        self.__seed = seed
