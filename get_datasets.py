import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

MAX_LENGTH = 150
BUFFER_SIZE = 1000
BATCH_SIZE = 128


def build_tokenizer(df):
    questions = df['questions'].dropna().to_list()
    corpus = questions+ df['answers'].dropna().to_list()
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(corpus, target_vocab_size=2**13)
    return tokenizer

def tf_encode(q_text, a_text, tokenizer):
    def encode(q_text, a_text,):# Add a start and end token to the input and target.
        q_text = [tokenizer.vocab_size] + tokenizer.encode(q_text.numpy()) + [tokenizer.vocab_size + 1]
        a_text = [tokenizer.vocab_size] + tokenizer.encode(a_text.numpy()) + [tokenizer.vocab_size + 1]
        return q_text, a_text
    # you can't .map this function directly: You need to wrap it in a tf.py_function.
    # The tf.py_function will pass regular tensors (with a value and a .numpy() method to access it), to the wrapped python function.
    result_q, result_a = tf.py_function(encode, [q_text, a_text], [tf.int64, tf.int64])
    result_q.set_shape([None])
    result_a.set_shape([None])
    return result_q, result_a

def build_datasets():
    # drop examples with a length of over 150 tokens.

    def filter_max_length(x, y, max_length=MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)

    df = pd.read_csv('data/data.csv').dropna()
    tokenizer = build_tokenizer(df)
    def encode_map(x, y):
        return tf_encode(x, y, tokenizer)
    train_X, test_X, train_y, test_y = train_test_split(df['questions'].to_numpy(), df['answers'].to_numpy())
    # print(train_X)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    val_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))

    train_dataset = train_dataset.map(encode_map)
    train_dataset = train_dataset.filter(filter_max_length)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)

    val_dataset = val_dataset.map(encode_map)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)

    return train_dataset, val_dataset, tokenizer


if __name__ == '__main__':
    train_dataset, val_dataset, _ = get_dataset()
    q_batch, a_batch = next(iter(val_dataset))
    print(q_batch, a_batch)

