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