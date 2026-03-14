Word Embeddings
============

Distributed representations (DR) of words (i.e., word embeddings) are used to capture semantic and syntactic regularities of the language by analyzing distributions of word relations within the textual data. Modeling methods generating DRs rely on the assumption that 'words that occur in similar contexts tend to have similar meanings' (distributional hypothesis) which stems from the nature of language itself. Due to their unsupervised nature, these modeling methods do not require any human judgement input to train, which allows researchers to train very large datasets in relatively low costs.

Traditional representations of words (i.e., one-hot vectors) are based on word-word (W x W) co-occurrence sparse matrices where W is the number of distinct words in the corpus. On the other hand, distributed word representations (DRs) (i.e., word embeddings) are word-context (W x C) dense matrices where C < W and C is the number of context dimensions which are determined by underlying model assumptions. Dense representations are arguably better at capturing generalized information and more resistant to overfitting due to context vectors representing shared properties of words. DRs are real valued vectors where each context can be considered as a continuous feature of a word. Due to their ability to represent abstract features of a word, DRs are considered as reusable across higher level tasks in ease, even if they are trained with totally different datasets.

Prediction based DR models gained much attention after Mikolov et al.’s neural network based SkipGram model in 2013. The secret behind the prediction based models is simple: never build a sparse matrix at all. Prediction based models construct dense matrix representations directly instead of reducing sparse ones to dense ones. These models are trained like any other supervised learning task by giving lots of positive and negative samples without adding any human supervision costs. Aim of these models is to maximize the probability of each context c with the same distributional assumptions on word-context co-occurrences, similar to count based models.

SkipGram is a prediction based distributional semantic model (DSM) consisting of a shallow neural network architecture inspired from neural language modeling (LM) intuitions. It is commonly known for its open-source implementation library word2vec. SkipGram acts like a log-linear classifier maximizing the prediction of the surrounding words of a word within a context (center window). Probabilistic word and sentence prediction by local neighbors of a word has been successfully applied on LM tasks under Markov assumption. SkipGram leverages the same idea by considering the words within the window as positive and negative instances and learning weights (for k contexts) which maximizes word predictions. In the training process, each word vector starts as a random vector, and then iteratively shifts to the neighboring vector.

Video Lectures
============

[<img src=https://github.com/StarlangSoftware/WordToVec/blob/master/video1.jpg width="50%">](https://youtu.be/7zaTW8dH_QMc)[<img src=https://github.com/StarlangSoftware/WordToVec/blob/master/video2.jpg width="50%">](https://youtu.be/4YvBJ_p8HRc)[<img src=https://github.com/StarlangSoftware/WordToVec/blob/master/video3.jpg width="50%">](https://youtu.be/Hh-MM_rSWeo)

For Developers
============

You can also see [Cython](https://github.com/starlangsoftware/WordToVec-Cy), [Java](https://github.com/starlangsoftware/WordToVec), [C](https://github.com/starlangsoftware/WordToVec-C), [C++](https://github.com/starlangsoftware/WordToVec-CPP), [Swift](https://github.com/starlangsoftware/WordToVec-Swift), 
[Js](https://github.com/starlangsoftware/WordToVec-Js) or [C#](https://github.com/starlangsoftware/WordToVec-CS) repository.

## Requirements

* [Python 3.7 or higher](#python)
* [Git](#git)

### Python 

To check if you have a compatible version of Python installed, use the following command:

    python -V
    
You can find the latest version of Python [here](https://www.python.org/downloads/).

### Git

Install the [latest version of Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

## Pip Install

	pip3 install NlpToolkit-WordToVec
	
## Download Code

In order to work on code, create a fork from GitHub page. 
Use Git for cloning the code to your local or below line for Ubuntu:

	git clone <your-fork-git-link>

A directory called WordToVec will be created. Or you can use below link for exploring the code:

	git clone https://github.com/starlangsoftware/WordToVec-Py.git

## Open project with Pycharm IDE

Steps for opening the cloned project:

* Start IDE
* Select **File | Open** from main menu
* Choose `WordToVec-Py` file
* Select open as project option
* Couple of seconds, dependencies will be downloaded. 

Detailed Description
============

To initialize artificial neural network:

	NeuralNetwork(self, corpus: Corpus, parameter: WordToVecParameter)

To train neural network:

	train(self) -> VectorizedDictionary

# Cite

	@inproceedings{ercan-yildiz-2018-anlamver,
    	title = "{A}nlam{V}er: Semantic Model Evaluation Dataset for {T}urkish - Word Similarity and Relatedness",
    	author = {Ercan, G{\"o}khan  and
      	Y{\i}ld{\i}z, Olcay Taner},
    	booktitle = "Proceedings of the 27th International Conference on Computational Linguistics",
    	month = aug,
    	year = "2018",
    	address = "Santa Fe, New Mexico, USA",
    	publisher = "Association for Computational Linguistics",
    	url = "https://www.aclweb.org/anthology/C18-1323",
    	pages = "3819--3836",
	}

For Contibutors
============

### Setup.py file
1. Do not forget to set package list. All subfolders should be added to the package list.
```
    packages=['Classification', 'Classification.Model', 'Classification.Model.DecisionTree',
              'Classification.Model.Ensemble', 'Classification.Model.NeuralNetwork',
              'Classification.Model.NonParametric', 'Classification.Model.Parametric',
              'Classification.Filter', 'Classification.DataSet', 'Classification.Instance', 'Classification.Attribute',
              'Classification.Parameter', 'Classification.Experiment',
              'Classification.Performance', 'Classification.InstanceList', 'Classification.DistanceMetric',
              'Classification.StatisticalTest', 'Classification.FeatureSelection'],
```
2. Package name should be lowercase and only may include _ character.
```
    name='nlptoolkit_math',
```

### Python files
1. Do not forget to comment each function.
```
    def __broadcast_shape(self, shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Determines the broadcasted shape of two tensors.

        :param shape1: Tuple representing the first tensor shape.
        :param shape2: Tuple representing the second tensor shape.
        :return: Tuple representing the broadcasted shape.
        """
```
2. Function names should follow caml case.
```
    def addItem(self, item: str):
```
3. Local variables should follow snake case.
```
	det = 1.0
	copy_of_matrix = copy.deepcopy(self)
```
4. Class variables should be declared in each file.
```
class Eigenvector(Vector):
    eigenvalue: float
```
5. Variable types should be defined for function parameters and class variables.
```
    def getIndex(self, item: str) -> int:
```
6. For abstract methods, use ABC package and declare them with @abstractmethod.
```
    @abstractmethod
    def train(self, train_set: list[Tensor]):
        pass
```
7. For private methods, use __ as prefix in their names.
```
    def __infer_shape(self, data: Union[List, List[List], List[List[List]]]) -> Tuple[int, ...]:
```
8. For private class variables, use __ as prefix in their names.
```
class Matrix(object):
    __row: int
    __col: int
    __values: list[list[float]]
```
9. Write \_\_repr\_\_ class methods as toString methods
10. Write getter and setter class methods.
```
    def getOptimizer(self) -> Optimizer:
        return self.optimizer
    def setValue(self, value: Optional[Tensor]) -> None:
        self._value = value
```
11. If there are multiple constructors for a class, define them as constructor1, constructor2, ..., then from the original constructor call these methods.
```
    def constructor1(self):
        self.__values = []
        self.__size = 0

    def constructor2(self, values: list):
        self.__values = values.copy()
        self.__size = len(values)

    def __init__(self,
                 valuesOrSize=None,
                 initial=None):
        if valuesOrSize is None:
            self.constructor1()
        elif isinstance(valuesOrSize, list):
            self.constructor2(valuesOrSize)
```
12. Extend test classes from unittest and use separate unit test methods.
```
class TensorTest(unittest.TestCase):

    def test_inferred_shape(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual((2, 2), a.getShape())

    def test_shape(self):
        a = Tensor([1.0, 2.0, 3.0])
        self.assertEqual((3, ), a.getShape())
```
13. Enumerated types should be used when necessary as enum classes.
```
class AttributeType(Enum):
    """
    Continuous Attribute
    """
    CONTINUOUS = auto()
    """
    Discrete Attribute
    """
    DISCRETE = auto()
```
