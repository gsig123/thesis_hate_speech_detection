# Multilingual Hate Speech Detection 
### Master's Thesis // IT University of Copenhagen 

## Project Setup 
**Project Dependencies:**
- **Python 3** :snake:

**Setting up a Python 3 environment with [venv](https://docs.python.org/3/library/venv.html#module-venv):**
```console
# Create the environment
$ python3 -m venv ~/PythonEnv/thesis
# Activate it 
$ . ~/PythonEnv/thesis/bin/activate
```

**Installing Requirements:**
```console
$ pip install -r requirements.txt
```

## Preprocessing
- `./src/preprocess/`
- A directory containing classes needed for preprocessing the data for the different classifiers.

### Base Class
- `data_prep_base.py` 
- An abstract base class that implements methods to create `Pandas` `DataFrame` from `tsv` and `csv` files. 
- Also Implements some methods to get `X` and `y`, split the data into `train` and `test` sets, and more. 
- Classes inheriting from the base class must implement the `init` method. 

### Automated Hate Speech Detection and the Promblem of Offensive Language Data Prep
- `data_prep_hsaofl.py` 
- Implements methods to get the dataset ready for the model introduced [here](https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/src/Automated%20Hate%20Speech%20Detection%20and%20the%20Problem%20of%20Offensive%20Language.ipynb) - This includes extracting features etc. 
- the `get_X_y_feature_names` method takes in a dataset as a `Pandas` `DataFrame` and returns `X`, `y`, and `feature_names`, ready as inputs for the corresponding classifier.

## Classifiers
- `./src/classifiers/`
- A directory containing all the classifiers implemented. 

### Base Class
- `classifier_base.py`
- An abstract base class that implements methods to compute `confusion_matrix`, `f1_score`, `recall`, `precision` `accuracy` and a method to plot the `confusion_matrix`. 
- Also requires all classes that inherit from it to implement `init`, `fit` and `predict` methods. 

### Automated Hate Speech Detection and the Promblem of Offensive Language Classifier
- `classifier_hsaofl.py` 
- Implements the model introduced [here](https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/src/Automated%20Hate%20Speech%20Detection%20and%20the%20Problem%20of%20Offensive%20Language.ipynb)
- A script showing how it can be used can be found in `./src/scripts/test_hsaofl_classifier.py`.

## Refernces 
### Automated Hate Speech Detection and the Problem of Offensive Language
- [Link](https://github.com/t-davidson/hate-speech-and-offensive-language/)
- Citation: 
```
@inproceedings{hateoffensive,
  title = {Automated Hate Speech Detection and the Problem of Offensive Language},
  author = {Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar}, 
  booktitle = {Proceedings of the 11th International AAAI Conference on Web and Social Media},
  series = {ICWSM '17},
  year = {2017},
  location = {Montreal, Canada},
  pages = {512-515}
  }
```

## Contributors 

- [Gudbjartur ( Bjartur ) Sigurbergsson](sigurberg.son@gmail.com)
- [Leon Derczynski](ld@itu.dk) 