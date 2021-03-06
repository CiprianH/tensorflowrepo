import numpy as np

import pandas as pd

import sklearn

import sklearn.preprocessing

import tensorflow as tf

from matplotlib import pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

tf.logging.log(tf.logging.INFO, "Tensorflow version " + tf.__version__)


DATA_PATH = "prices-split-adjusted.csv"

df = pd.read_csv(DATA_PATH, index_col=0)


SYMBOL_NAME='GOOG'

df_stock = df[df.symbol == SYMBOL_NAME].copy()


df_stock.drop(['symbol'], 1, inplace=True)

df_stock.drop(['volume'], 1, inplace=True)


def scale_data(df):

    min_max_scaler = sklearn.preprocessing.MinMaxScaler()

    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))

    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))

    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))

    df['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))


    return df


df_stock_norm = df_stock.copy()

df = scale_data(df_stock_norm)


SEQLEN = 20

df_input = df

df_target = df.shift(-1)

df_input = df_input[:-2].values

df_target = df_target[:-2].values

X = np.reshape(df_input, (-1, SEQLEN, 4))

Y = np.reshape(df_target, (-1, SEQLEN, 4))


train_split = 0.8

num_data = X.shape[0]

num_train = int(train_split * num_data)

x_train = X[0:num_train]

y_train = Y[0:num_train]

y_test = Y[num_train:]

x_test = X[num_train:]


def train_input():

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    dataset = dataset.repeat()

    dataset = dataset.shuffle(SHUFFLE_SIZE)

    dataset = dataset.batch(BATCHSIZE)

    samples, labels = dataset.make_one_shot_iterator().get_next()

    return samples, labels


def test_input():

    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    dataset = dataset.repeat(1)

    dataset = dataset.batch(BATCHSIZE)

    samples, labels = dataset.make_one_shot_iterator().get_next()

    return samples, labels


RNN_CELLSIZE = 80

N_LAYERS = 2

DROPOUT_PKEEP = 0.7

def model_rnn_fn(features, labels, mode):

    batchsize = tf.shape(features)[0]

    seqlen = tf.shape(features)[1]

    cells = [tf.nn.rnn_cell.GRUCell(RNN_CELLSIZE) for _ in range(N_LAYERS)]

    cells[:-1] = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = DROPOUT_PKEEP) for cell in cells[:-1]]

    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=False)


    Yn, H = tf.nn.dynamic_rnn(cell, features, dtype=tf.float64)


    Yn = tf.reshape(Yn, [batchsize * seqlen, RNN_CELLSIZE])

 

    Yr = tf.layers.dense(Yn, 4)  # Yr l[BATCHSIZE*SEQLEN, 1]

    Yr = tf.reshape(Yr, [batchsize, seqlen, 4])  # Yr [BATCHSIZE, SEQLEN, 1]

 

    Yout = Yr[:, -1, :]  # Last output Yout [BATCHSIZE, 1]

 

    loss = train_op = None

    if mode != tf.estimator.ModeKeys.PREDICT:

        loss = tf.losses.mean_squared_error(Yr, labels)  # la  bels[BATCHSIZE, SEQLEN, 1]

        lr = 0.001

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    

        train_op = tf.contrib.training.create_train_op(loss, optimizer)

 

    return tf.estimator.EstimatorSpec(

        mode=mode,

        predictions={"Yout": Yout},

        loss=loss,

        train_op=train_op

    )



training_config = tf.estimator.RunConfig(model_dir="./output")


estimator = tf.estimator.Estimator(model_fn=model_rnn_fn, config=training_config)


SHUFFLE_SIZE = 1

BATCHSIZE = 50

estimator.train(input_fn=train_input, steps=10000)


results = estimator.predict(test_input)

Yout_ = [result["Yout"] for result in results]

predict = np.array(Yout_)

actual = y_test[:, -1]

colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.plot(actual[:, 0], label="Actual Values", color='red')

plt.plot(predict[:, 0], label="Predicted Values", color='green')

plt.title('stock')

plt.xlabel('time [days]')

plt.ylabel('normalized price')

plt.legend(loc='best')

plt.show()
