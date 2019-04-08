# Multilingual Hate Speech Detection 
### Master's Thesis // IT University of Copenhagen 

## Table of Contents

1. [Project Setup](#project-setup)
2. [Preprocessing](#preprocessing)
3. [Features](#feature-extraction)
4. [Classifiers](#classifiers)
4. [References](#references)
5. [Contributors](#contributors)

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
- The dataset this is designed for can be found in `./data/raw/HateSpeechAndOffensiveLanguage/labeled_data.csv`.

### OffensEval
- `data_prep_offenseval.py`
- Implements functionality to read in a file and turn it into `X`, `y_a`, `y_b`, `y_c`, corresponding to the different subtasks as they are defined in the `OffensEval` task. 

## Feature Extraction
- `./src/feature_extraction/`
- A directory containing modules for extracting different features from the dataset.
- `tokenize.py`: Tokenize a sentence. Excludes stopwords and punctuation. 
- `bag_of_words.py`: Create BOW representation from list of sentences. 
- `tfidf_from_bow.py`: Transform a BOW representation to a `tf-idf` representation. 
- `sentiment_score_english.py`: Get a sentiment score for a sentence. -1 is negative, 0 is neutral, +1 is positive. 
- `w2v_embeddings.py`: Create a W2V Model based on the input corpus. Returns embeddings for the input corpus. 
- `w2i.py`: Takes in a list of sentences and returns a list of lists of integers, where each
integer represent a word. The default behaviour is to pad the sequence so all sequences 
have the same length once outputted. 

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

### Bi LSTM
- `classifier_bi_lstm.py`
- BI-LSTM + Multilayer Perceptron Classifier

## FastText

### Create Word Embeddings from vocabulary using FastText
```sh
sudo ./fastText/fasttext skipgram -dim 300 -input ./data/processed/OffensEval2019/start-kit/training-v1/fasttext-dataset.txt -output ./models/fast_text/OffensEval_EN_300d
```

### Use a FastText model to get embeddings for a word
```python
from pyfasttext import FastText

model_path = "./models/fast_text/OffensEval_EN_300d.bin"
model = FastText(model_path)
emb = model["@USER"]
```


## References 

### Links
- [Automated Hate Speech Detection and the Problem of Offensive Language](https://github.com/t-davidson/hate-speech-and-offensive-language/)
- [OffensEval 2019 ](https://competitions.codalab.org/competitions/20011#learn_the_details)
- [Deep Learning for Hate Speech Detection](https://github.com/pinkeshbadjatiya/twitter-hatespeech)
- [FastText Pretrained Embeddings](https://fasttext.cc/docs/en/crawl-vectors.html)

### Citations
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
```
@inproceedings{grave2018learning,
  title={Learning Word Vectors for 157 Languages},
  author={Grave, Edouard and Bojanowski, Piotr and Gupta, Prakhar and Joulin, Armand and Mikolov, Tomas},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
```

## Contributors 

- [Gudbjartur ( Bjartur ) Sigurbergsson](sigurberg.son@gmail.com)
- [Leon Derczynski](ld@itu.dk) 