import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np

text = open("CollegeEssays.txt", "rb").read().decode("utf8")

print(len(text))

text = text.replace(".", "").replace(",", "").replace("!", "").replace("?", "").replace('“', '').replace("”", '').replace('"', '').lower().split(" ") # Dataset too small to grasp punctuation while splitting by words
vocab = sorted(set(text))

ids_from_words = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None
)

words_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_words.get_vocabulary(), invert=True, mask_token=None)


def text_from_ids(ids):
  return tf.strings.reduce_join(words_from_ids(ids), axis=-1)


ids = ids_from_words(text)
# print(ids_from_words.vocabulary_size())
words_dataset = tf.data.Dataset.from_tensor_slices(ids)

# for id in words_dataset.take(5):
#     print(words_from_ids(id.numpy()))


def get_windowed_ds(window_size, ds, BATCH_SIZE):
    window_ds = ds.window(window_size + 1, 1, drop_remainder=True)
    window_ds = window_ds.flat_map(lambda x: x.batch(window_size + 1))
    window_ds = window_ds.map(lambda window: (window[:-1], window[-1]))
    window_ds = window_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return window_ds


window_ds = get_windowed_ds(20, words_dataset, 128)
# for window in window_ds.take(1):
#     print(window)


vocab_size = ids_from_words.vocabulary_size()
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 256),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(vocab_size, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy()
)

# model.fit(
#     window_ds, epochs=50, callbacks=[tf.keras.callbacks.TensorBoard()]
# )
# model.save("model.h5")

model = tf.keras.models.load_model("model.h5")
output = []
s = "I would like to begin".split(" ")
original_s = "I would like to begin"

for i in range(75):
    prediction = model.predict([ids_from_words(np.expand_dims(s, axis=0))])
    prediction_word = words_from_ids(np.argmax(prediction)).numpy().decode("utf8")
    output.append(prediction_word)
    s = s[1:]
    s.append(prediction_word)

print(original_s + " " + " ".join(output))