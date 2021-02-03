For Developers
============

You can also see [Java](https://github.com/starlangsoftware/WordToVec), [C++](https://github.com/starlangsoftware/WordToVec-CPP), [Swift](https://github.com/starlangsoftware/WordToVec-Swift), or [C#](https://github.com/starlangsoftware/WordToVec-CS) repository.

## Requirements

* [Python 3.7 or higher](#python)
* [Git](#git)

### Python 

To check if you have a compatible version of Python installed, use the following command:

    python -V
    
You can find the latest version of Python [here](https://www.python.org/downloads/).

### Git

Install the [latest version of Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

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
