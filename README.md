# suffix-lemmatizer

Suffix-lemmatizer is a lemmatizer for Estonian language, which handles both in- and out-of-vocabulary (OOV) words. OOV issue is addressed by generating candidate lemmas based on suffix transformations and ranking them using a statistical model.

Suffix-lemmatizer works with Python 2.7.

## Installation
```
git clone https://github.com/estnltk/suffix-lemmatizer.git
cd suffix-lemmatizer
python setup.py install
```

## Usage
```python
  from suffix_lemmatizer import SuffixLemmatizer
  sl = SuffixLemmatizer()
  lemma = sl('metsast')
  print(lemma)
  >>> 'mets'
```
