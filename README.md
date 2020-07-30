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

