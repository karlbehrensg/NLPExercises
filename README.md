# Natural Lenguaje Procesing

## Word based encoding

To encoding the words for a sentence we use `Tokenizer`. It will encoding each word of our list of sentences, and `num_words` parameter will take the top words by volume and just encode those.

To encoding our list we use `fit_on_text` from `Tokenizer`

We can get a dictionary with the key values pair with `word_index`, and the value is the token for that word.

```python
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!'
]

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

# Result
{'love': 1, 'my': 2, 'i': 3, 'dog': 4, 'cat': 5, 'you': 6}
```

## Text to sequence and padding

We can get a sequence of the encoding phrase using `text_to_sequences`. If we try pass word without encoding, this words will be ignore.

With a property from `Tokenizer` call `oov_token` (out of vocabulary) we can replace the ignores word with a specific token. The better way to use this token is use a distinct and unique to not confused with a real word.

At last we need to padding our sentences with `pad_sequences`, this will give the same dimensionality to each sentences. With `padding` we define if we put 0 before (default) or after ('post') the sentences, to define the max length we use `maxlen` and with `truncating` we define if will be truncating taken the tail (default) loosing the beginning of the sentences or take the head ('post').

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, maxlen=5)
print("\nWord Index = " , word_index)
print("\nSequences = " , sequences)
print("\nPadded Sequences:")
print(padded)


# Try with words that the tokenizer wasn't fit to
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)

padded = pad_sequences(test_seq, maxlen=10)
print("\nPadded Test Sequence: ")
print(padded)


# Results
Word Index =  {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}

Sequences =  [[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]

Padded Sequences:
[[ 0  5  3  2  4]
 [ 0  5  3  2  7]
 [ 0  6  3  2  4]
 [ 9  2  4 10 11]]

Test Sequence =  [[5, 1, 3, 2, 4], [2, 4, 1, 2, 1]]

Padded Test Sequence: 
[[0 0 0 0 0 5 1 3 2 4]
 [0 0 0 0 0 2 4 1 2 1]]
```

## Loading a JSON file.

```python
import json

with open('sarcasm.json', 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
```

## Tensorflow datasets

```python
import tensorflow_datasets as tfds

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)


"""Separate the data in train and test sets"""
train_data, test_data = imdb['train'], imdb['test']

"""Separating sentences and labels"""
import numpy as np

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s, l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())

"""Labels are expected to be Numpy arrays."""
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

"""Embedding the words"""
vocab_size = 10000
embedding_dim =  16
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV>'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

"""Define the model"""
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

"""Compile the model"""
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

"""Dimension weights layers"""
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)
# Result (10000, 16) 

"""Reverse word index"""
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

"""Write metadata auto files"""
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + '\n')
    out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
out_v.close()
out_m.close()
```

## LSTM
```python
import tensorflow as tf

"""A basic model of LSTM"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.Biderectional(tf.keras.layers.LSTM(64)), # With Biderectional layer we double the output shape.
    tf.keras.Dense(64, activation='relu'),
    tf.keras.Dense(1, activation='sigmoid')
])

"""Model with more than 1 biderectional layer"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.Biderectional(tf.keras.layers.LSTM(64), return_sequences=True), # Always the first in a sequences of biderectional layers will have return_sequences=True
    tf.keras.Biderectional(tf.keras.layers.LSTM(32)),
    tf.keras.Dense(64, activation='relu'),
    tf.keras.Dense(1, activation='sigmoid')
])
```