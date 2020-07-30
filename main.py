import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def act(z):
    return tf.exp(z)


def cube(z):
    return tf.pow(z, 3)


initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.9,
    staircase=True)


# read and manipulate data
df = pd.read_csv("tablecorona.csv")
dates = df['Date'].to_numpy(dtype=np.datetime64)
reldates = dates - dates[0]
reldates = reldates.astype(np.int)
Deaths = df['Deaths'].to_numpy(dtype=np.int32)
Recoveries = df['Recoveries'].to_numpy(dtype=np.int32)
Total = df['Total confirmed'].to_numpy(dtype=np.int32)
x = reldates
y = Total          #estaminating total cases here
mean = np.mean(x)
std = np.std(x)
x = x - mean
x = x / std
ymean = np.mean(y).astype(np.int)
ystd = np.std(y)
y = y -mean
y =y/ystd
x = tf.constant(x)
y = tf.constant(y)
input_layer = tf.keras.Input(shape=(1), dtype='float32', name='input')
midlay = tf.keras.layers.Dense(1, activation=act)(input_layer)
midla2 = tf.keras.layers.Dense(1, activation=tf.square)(input_layer)
midlay3 = tf.keras.layers.Dense(1, activation=cube)(input_layer)
merged = tf.keras.layers.concatenate([midlay, midla2, midlay3], axis=1)
output = tf.keras.layers.Dense(1, name='output')(merged)
model = tf.keras.Model(inputs=[input_layer], outputs=[output])
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule),
    loss=tf.keras.losses.mean_squared_error,
)
hist = model.fit(x, y, epochs=1000, batch_size=154)
pred = model.predict(x)
plt.plot(x,y)
plt.plot(x,pred)
plt.show()
