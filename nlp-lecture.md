
# Natural Language Processing

J.F. Omhover, with thanks to Mari Pierce-Quinonez for some great enhancements.

Updated for Python 3.6 by Miles Erickson

### Requirements

You need to install the `nltk` module:

```
conda install nltk
```

This module will need corporas that you need to download. This can take a very long time, for simplicity here's the minimal corporas for this lecture. In a terminal, open `ipython` and type:

```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_treebank_pos_tagger')
```

### General Introduction

Natural Language Processing is a subfield of machine learning focused on making sense of text. Text is inherently unstructured and has all sorts of tricks required for converting (vectorizing) text into a format that a machine learning algorithm can interpret. It is called Processing for a reason - most of what we'll be covering during this morning session are Data Processing operations that make it possible to plug test into other ML algorythms.

### Overview of nlp

Natural language processing is concerned with understanding text using computation. People working within the field are often concerned with:
- Information retrieval. How do you find a document or a particular fact within a document?
- Document classification. What is the document about amongst mutually exclusive categories?
- Machine translation. How do you write an English phrase in Chinese? Think of Google translate.
- Sentiment analysis. Was a product review positive or negative?
Natural language processing is a huge field and we will just touch on some of the concepts.

### Objectives

- Name and describe the steps necessary for processing text in machine learning.
- Implement a Natural Language Processing pipeline.
- Explain the cosine similarity measure and why it is used in NLP.

# Text Featurization part 1 : Bags of Words

This Walkthrough will lead us from raw documents to bag-of-words representations using **Natural Language Processing** functions.

In our case, this walkthrough is a preliminary step of a pipeline for **indexing** documents.

The ultimate goal of **indexing** is to create a **signature** (vector) for each document.

This **signature** will be used for relating documents one to the other (and find out similar clusters of documents), or for mining underlying relations between concepts.

<img src="img/pipeline-walkthrough1.png" width="70%"/>

## 0. Text sources and possible text mining inputs


```python
paragraph = "My mother drove me to the airport with the windows rolled down. It was seventy-five degrees in Phoenix, the sky a perfect, cloudless blue. I was wearing my favorite shirt – sleeveless, white eyelet lace; I was wearing it as a farewell gesture. My carry-on item was a parka. In the Olympic Peninsula of northwest Washington State, a small town named Forks exists under a near-constant cover of clouds. It rains on this inconsequential town more than any other place in the United States of America. It was from this town and its gloomy, omnipresent shade that my mother escaped with me when I was only a few months old. It was in this town that I’d been compelled to spend a month every summer until I was fourteen. That was the year I finally put my foot down; these past three summers, my dad, Charlie, vacationed with me in California for two weeks instead."

paragraph
```


```python
import unicodedata

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()

input_string = remove_accents(paragraph)
```

# 1. Creating bag-of-words for each document

## 1.1. Tokenize document

**"Tokenize"** means creating "tokens" which are atomic units of the text. These tokens are usually words we extract from the document by splitting it (using punctuations as a separator). We can also consider sentences as tokens (and words as sub-tokens of sentences).

### nltk.tokenize.sent_tokenize


```python
from nltk.tokenize import sent_tokenize

sent_tokens = sent_tokenize(input_string)

sent_tokens
```

### nltk.tokenize.word_tokenize


```python
from nltk.tokenize import word_tokenize

tokens = [sent for sent in map(word_tokenize, sent_tokens)]

list(enumerate(tokens))
```

### lower


```python
import string

tokens_lower = [[word.lower() for word in sent]
                 for sent in tokens]
```

## 1.2. Filtering stopwords (and punctuation)

**Stopwords** are words that should be stopped at this step because they do not carry much information about the actual meaning of the document. Usually, they are "common" words you use. You can find lists of such **stopwords** online, or embedded within the NLTK library.

### Using your own stop list


```python
from nltk.corpus import stopwords

stopwords_ = set(stopwords.words('english'))

print("--- stopwords in english: {}".format(stopwords_))
```


```python
# list found at http://www.textfixer.com/resources/common-english-words.txt
# 'not' has been removed (do you know why ?)

stopwords_ = "a,able,about,across,after,all,almost,also,am,among,an,and,any,\
are,as,at,be,because,been,but,by,can,could,dear,did,do,does,either,\
else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,\
how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,\
me,might,most,must,my,neither,no,of,off,often,on,only,or,other,our,\
own,rather,said,say,says,she,should,since,so,some,than,that,the,their,\
them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,\
what,when,where,which,while,who,whom,why,will,with,would,yet,you,your]".split(',')

print("--- stopwords in english: {}".format(stopwords_))
```

We also need to filter punctuation tokens: tokens made of punctuation marks. We can find a list of those punctuations in string.punctuation.


```python
import string

punctuation_ = set(string.punctuation)
print("--- punctuation: {}".format(string.punctuation))

def filter_tokens(sent):
    return([w for w in sent if not w in stopwords_ and not w in punctuation_])

tokens_filtered = list(map(filter_tokens, tokens_lower))

for sent in tokens_filtered:
    print("--- sentence tokens: {}".format(sent))
```

## 1.3. Stemming and lemmatization

**Stemming** means reducing each word to a **stem**. That is, reducing each word in all its diversity to a root common to all its variants.


```python
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

stemmer_porter = PorterStemmer()
tokens_stemporter = [list(map(stemmer_porter.stem, sent)) for sent in tokens_filtered]
print("--- sentence tokens (porter): {}".format(tokens_stemporter[0]))

stemmer_snowball = SnowballStemmer('english')
tokens_stemsnowball = [list(map(stemmer_snowball.stem, sent)) for sent in tokens_filtered]
print("--- sentence tokens (snowball): {}".format(tokens_stemsnowball[0]))
```

## 1.4. N-Grams

<span style="color:red">To capture sequences of tokens</span>


```python
from nltk.util import ngrams

list(ngrams(tokens_stemsnowball[0],2))
```


```python
from nltk.util import ngrams

def join_sent_ngrams(input_tokens, n):
    # first add the 1-gram tokens
    ret_list = list(input_tokens)
    
    #then for each n
    for i in range(2,n+1):
        # add each n-grams to the list
        ret_list.extend(['-'.join(tgram) for tgram in ngrams(input_tokens, i)])
    
    return(ret_list)

tokens_ngrams = list(map(lambda x : join_sent_ngrams(x, 3), tokens_stemporter))

print("--- sentence tokens: {}".format(tokens_ngrams[0]))
```

## 1.5. Part-of-Speech tagging

This is an alternative process that relies on machine learning to tag each word in a sentence with its function. In libraries such as NLTK, there are embedded tools to do that. Tags detected depend on the corpus used for training. In NLTK, the function `nltk.pos_tag()` uses the [Penn Treebank](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).

### nltk.pos_tag


```python
from nltk import pos_tag

sent_tags = list(map(pos_tag, tokens))

for sent in sent_tags:
    print("--- sentence tags: {}".format(sent))
```

Let's filter verbs !


```python
for sent in sent_tags:
    tags_filtered = [t for t in sent if t[1].startswith('VB')]
    print("--- verbs:\n{}".format(tags_filtered))
```


```python
from nltk import RegexpParser

grammar = r"""
  NPB: {<DT|PP\$>?<JJ|NN|,>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
  V2V: {<V.*> <TO> <V.*>}
"""

cp = RegexpParser(grammar)
result = cp.parse(sent_tags[1])

#print result

for sent in sent_tags:
    tree = cp.parse(sent)
    for subtree in tree.subtrees():
        if subtree.label() == 'NPB': print(subtree)
        if subtree.label() == 'V2V': print(subtree)
```

# Text Featurization part 2 : Indexing Bag-of-Words into a vector table

This Walkthrough will lead us from bag-of-words representations of documents to **vector signatures** (indexes) using the **TF-IDF** formula.

The ultimate goal of **indexing** is to create a **vector representation** (signature) for each document. This vector representation will be used for:

- mine the features that can caracterize classes of documents (supervised learning using **labels**)
- mine the documents that have similar features to establish trends (unsupervised learning).

To do that, we need:
- a fixed number of features
- a quantitative value for each feature.

The number of features is given by the vocabulary over the corpus: the set of all possible words (tokens) found in all documents.

The quantitative value is given, for each doc, by counting the occurences of each of these words in the doc and by using a TF-IDF formula.

<img src="img/pipeline-walkthrough2.png" width="70%"/>

## 0. Loading some input data from the Amazon Reviews

To try this indexing walkthrough, we will get 5 reviews from the Amazon Reviews dataset. We will apply a function for extracting bag-of-words representations from these 5 documents.


```python
import os               # for environ variables in Part 3
from nlp_pipeline import extract_bow_from_raw_text
import json

docs = []
with open('./reviews.json', 'r') as data_file:    
    for line in data_file:
        docs.append(json.loads(line))

# extracting bows
bows = list(map(lambda row: extract_bow_from_raw_text(row['reviewText']), docs))

# displaying bows
for i in range(len(docs)):
    print("\n--- review: {}".format(docs[i]['reviewText']))
    print("--- bow: {}".format(bows[i]))
```

# 1. Indexing Bag of Words into a Vector Matrix using Term Frequency / Inverse Document Frequency
The ultimate goal of indexing is to create a vector representation (signature) for each document. This vector representation will be used for:
mine the features that can caracterize classes of documents (supervised learning using labels)
mine the documents that have similar features to establish trends (unsupervised learning).
To do that, we need:
- a fixed number of features
- a quantitative value for each feature.

The number of features is given by the vocabulary over the corpus: the set of all possible words (tokens) found in all documents.

The quantitative value is given, for each doc, by counting the occurences of each of these words in the doc and by using a TF-IDF formula.

## 1.1 Term Frequency

The number of times a term occurs in a specific document:

$tf(term,document) = \# \ of \ times \ a \ term \ appears \ in \ a \ document$


```python
from collections import Counter

# term occurence = counting distinct words in each bag
term_occ = list(map(lambda bow : Counter(bow), bows))

# term frequency = occurences over length of bag
term_freq = list()
for i in range(len(docs)):
    term_freq.append( {k: (v / float(len(bows[i])))
                       for k, v in term_occ[i].items()} )

# displaying occurences
for i in range(len(docs)):
    print("\n--- review: {}".format(docs[i]['reviewText']))
    print("--- bow: {}".format(bows[i]))
    print("--- term_occ: {}".format(term_occ[i]))
    print("--- term_freq: {}".format(term_freq[i]))
```

## 1.2. Obtaining document frequencies

$df(term,corpus) = \frac{ \# \ of \ documents \ that \ contain \ a \ term}{ \# \ of \ documents \ in \ the \ corpus}$



```python
# document occurence = number of documents having this word
# term frequency = occurences over length of bag

doc_occ = Counter( [word for bow in bows for word in set(bow)] )

# document frequency = occurences over length of corpus
doc_freq = {k: (v / float(len(docs)))
            for k, v in doc_occ.items()}

# displaying vocabulary
print("\n--- full vocabulary: {}".format(doc_occ))
print("\n--- doc freq: {}".format(doc_freq))
```

## 1.3 Creating the vocabulary for indexing


```python
# the minimum document frequency (in proportion of the length of the corpus)
min_df = 0.3

# filtering items to obtain the vocabulary
vocabulary = [ k for k,v in doc_freq.items() if v >= min_df ]

# print vocabulary
print ("-- vocabulary (len={}): {}".format(len(vocabulary),vocabulary))
```

## 1.4 the TFIDF vector

Words might show up a lot in individual documents, but their relevace is less important if they're in every document! We need to take into account words that show up everywhere and reduce their relative importance. The document frequency does exactly that:

$df(term,corpus) = \frac{ \# \ of \ documents \ that \ contain \ a \ term}{ \# \ of \ documents \ in \ the \ corpus}$

The inverse document frequency is defined in terms of the document frequency as

$idf(term,corpus) = \log{\frac{1}{df(term,corpus)}}$.


TF-IDF is an acronym for the product of two parts: the term frequency tf and what is called the inverse document frequency idf. The term frequency is just the counts in a term frequency vector. 

tf-idf $ = tf(term,document) * idf(term,corpus)$


```python
import numpy as np

# create a dense matrix of vectors for each document
# each vector has the length of the vocabulary
vectors = np.zeros((len(docs),len(vocabulary)))

# fill these vectors with tf-idf values
for i in range(len(docs)):
    for j in range(len(vocabulary)):
        term     = vocabulary[j]
        term_tf  = term_freq[i].get(term, 0.0)   # 0.0 if term not found in doc
        term_idf = np.log(1 + 1 / doc_freq[term]) # smooth formula
        vectors[i,j] = term_tf * term_idf

# displaying results
for i in range(len(docs)):
    print("\n--- review: {}".format(docs[i]['reviewText']))
    print("--- bow: {}".format(bows[i]))
    print("--- tfidf vector: {}".format( vectors[i] ) )
    print("--- tfidf sorted: {}".format( 
            sorted( zip(vocabulary,vectors[i]), key=lambda x:-x[1] )
         ))
```

## 1.5 Sklearn pipeline


```python
corpus = [row['reviewText'] for row in docs]
```


```python
from sklearn.feature_extraction.text import CountVectorizer

tf = CountVectorizer()

document_tf_matrix = tf.fit_transform(corpus).todense()

print(sorted(tf.vocabulary_))
print(document_tf_matrix)
```


```python
from math import log

def idf(frequency_matrix):
    df =  float(len(document_tf_matrix)) / sum(frequency_matrix > 0)
    return [log(i) for i in df.getA()[0]]
print(sorted(tf.vocabulary_))
print(idf(document_tf_matrix))
```


```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
document_tfidf_matrix = tfidf.fit_transform(corpus)
print(sorted(tfidf.vocabulary_))
print(document_tfidf_matrix.todense())
```

# Part 3 : Comparing two documents / Similarity Measures

## 3.1 Euclidean distance

We could try the Euclidean distance $||\vec{x}-\vec{y}||$  
What problems would we encounter with this? 

The euclidean distance goes up with the length of a document. Intuitively, duplicating each word in our bag of words generates a vector that points in exactly the same direction, however, the euclidean distance goes up. One solution is to normalize vectors before calculating the euclidean distance. Now increasing the length of a document does not change the Euclidean distance unless the direction of the term frequency vector changes. 

## 3.2 Cosine Similarity
Recall that for two vector $\vec{x}$ and $\vec{y}$ that $\vec{x} \cdot \vec{y} = ||\vec{x}|| ||\vec{y}|| \cos{\theta}$. And so,

$\frac{\vec{x} \cdot \vec{y} }{||\vec{x}|| ||\vec{y}||} = \cos{\theta}$

θ can only range from 0 to 90 degrees, because tf-idf vectors are non-negative. Therefore cos θ ranges from 0 to 1. Documents that are exactly identical will have cos θ = 1

