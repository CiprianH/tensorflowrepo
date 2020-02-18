import numpy as np

import pandas as pd

import sklearn

import sklearn.preprocessing

import tensorflow as tf

from matplotlib import pyplot as plt

#tf.logging.set_verbosity(tf.logging.INFO)

#tf.logging.log(tf.logging.INFO, "Tensorflow version " + tf.__version__)


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

    samples, labels = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    return samples, labels


def test_input():

    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    dataset = dataset.repeat(1)

    dataset = dataset.batch(BATCHSIZE)

    samples, labels = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    return samples, labels


#print(x_test[0:2])

x_input = np.array([[0.69273481, 0.68935482, 0.69136066, 0.69171589],
    [0.68599949, 0.69595, 0.69024042, 0.69260456],
    [0.68800505, 0.69705764, 0.68922191, 0.69572343],
    [0.69031143, 0.68534398, 0.69507798, 0.69184997],
    [0.68021692, 0.69474177, 0.69480645, 0.69209488],
    [0.6853644, 0.69145253, 0.69789563, 0.69503597],
    [0.69524173, 0.71583632, 0.70808007, 0.71691851],
    [0.71307428, 0.71399028, 0.70653547, 0.71275838],
    [0.70510223, 0.70160541, 0.7022241 , 0.70823257],
    [0.70384877, 0.69897067, 0.70483641, 0.70950697],
    [0.75351933, 0.74436513, 0.74391238, 0.7622599 ],
    [0.74329109, 0.74231772, 0.74055145, 0.7456593 ],
    [0.73701042, 0.73709865, 0.73948384, 0.74482089],
    [0.72980383, 0.73822299, 0.74117948, 0.73703201],
    [0.73279539, 0.74451611, 0.73876919, 0.73958921],
    [0.74131899, 0.73656162, 0.74253749, 0.74398248],
    [0.73246112, 0.7445665, 0.74377656, 0.74901295],
    [0.73055593, 0.7209547, 0.72461289, 0.74228887],
    [0.70435024, 0.66330966, 0.67272318, 0.70382252],
    [0.59274196, 0.62493002, 0.59246978, 0.63554228]])
    
y_input = np.array([[0.68599949, 0.69595, 0.69024042, 0.69260456],
    [0.68800505, 0.69705764, 0.68922191, 0.69572343],
    [0.69031143, 0.68534398, 0.69507798, 0.69184997],
    [0.68021692, 0.69474177, 0.69480645, 0.69209488],
    [0.6853644, 0.69145253, 0.69789563, 0.69503597],
    [0.69524173, 0.71583632, 0.70808007, 0.71691851],
    [0.71307428, 0.71399028, 0.70653547, 0.71275838],
    [0.70510223, 0.70160541, 0.7022241 , 0.70823257],
    [0.70384877, 0.69897067, 0.70483641, 0.70950697],
    [0.75351933, 0.74436513, 0.74391238, 0.7622599 ],
    [0.74329109, 0.74231772, 0.74055145, 0.7456593 ],
    [0.73701042, 0.73709865, 0.73948384, 0.74482089],
    [0.72980383, 0.73822299, 0.74117948, 0.73703201],
    [0.73279539, 0.74451611, 0.73876919, 0.73958921],
    [0.74131899, 0.73656162, 0.74253749, 0.74398248],
    [0.73246112, 0.7445665, 0.74377656, 0.74901295],
    [0.73055593, 0.7209547, 0.72461289, 0.74228887],
    [0.70435024, 0.66330966, 0.67272318, 0.70382252],
    [0.59274196, 0.62493002, 0.59246978, 0.63554228],
    [0.59274196, 0.62493002, 0.59246978, 0.63554228]])
    
def input_to_predict():

    dataset = tf.data.Dataset.from_tensor_slices((x_input, y_input))

    dataset = dataset.repeat(20)

    dataset = dataset.batch(BATCHSIZE)

    samples, labels = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    return samples, labels


RNN_CELLSIZE = 80

N_LAYERS = 2

DROPOUT_PKEEP = 0.7

def model_rnn_fn(features, labels, mode):

    batchsize = tf.shape(features)[0]

    seqlen = tf.shape(features)[1]

#    cells = [tf.nn.rnn_cell.GRUCell(RNN_CELLSIZE) for _ in range(N_LAYERS)]
    
    cells = [tf.compat.v1.nn.rnn_cell.GRUCell(RNN_CELLSIZE) for _ in range(N_LAYERS)]
    
#    cells[:-1] = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = DROPOUT_PKEEP) for cell in cells[:-1]]
    
#    cells[:-1] = [tf.nn.RNNCellDropoutWrapper(cell, output_keep_prob = DROPOUT_PKEEP) for cell in cells[:-1]]
  
    cells[:-1] = [tf.nn.RNNCellDropoutWrapper(cell, output_keep_prob = DROPOUT_PKEEP) for cell in cells[:-1]]
    
    cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=False)

#    Yn, H = tf.nn.dynamic_rnn(cell, features, dtype=tf.float64)
    
#    Yn, H = tf.compat.v1.nn.dynamic_rnn(cell, features, dtype=tf.float64)
    
    Yn, H = tf.compat.v1.nn.dynamic_rnn(cell, features, dtype=tf.float32)

    Yn = tf.reshape(Yn, [batchsize * seqlen, RNN_CELLSIZE])
 

#    Yr = tf.layers.dense(Yn, 4)  # Yr l[BATCHSIZE*SEQLEN, 1]
    
    Yr = tf.compat.v1.layers.dense(Yn, 4)

    Yr = tf.reshape(Yr, [batchsize, seqlen, 4])  # Yr [BATCHSIZE, SEQLEN, 1]

 

    Yout = Yr[:, -1, :]  # Last output Yout [BATCHSIZE, 1]

 

    loss = train_op = None

    if mode != tf.estimator.ModeKeys.PREDICT:

        loss = tf.losses.mean_squared_error(Yr, labels)  # la  bels[BATCHSIZE, SEQLEN, 1]

        lr = 0.001

#       optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        
        minimize_op = optimizer.minimize(
                loss,
                var_list=model.trainable_variables,
                global_step=tf.compat.v1.train.get_or_create_global_step())

#        train_op = tf.contrib.training.create_train_op(loss, optimizer)
        
        train_op = tf.group(loss, minimize_op)
 

    return tf.estimator.EstimatorSpec(

        mode=mode,

        predictions={"Yout": Yout},

        loss=loss,

        train_op=train_op

    )



training_config = tf.estimator.RunConfig(model_dir="./output")


#estimator = tf.estimator.Estimator(model_fn=model_rnn_fn, config=training_config)


SHUFFLE_SIZE = 1

BATCHSIZE = 50

#estimator.train(input_fn=train_input, steps=100)


#results = estimator.predict(test_input)

#Yout_ = [result["Yout"] for result in results]

#predict = np.array(Yout_)

#actual = y_test[:, -1]

#colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']

#plt.plot(actual[:, 0], label="Actual Values", color='red')

#plt.plot(predict[:, 0], label="Predicted Values", color='green')

#plt.title('stock')

#plt.xlabel('time [days]')

#plt.ylabel('normalized price')

#plt.legend(loc='best')

#plt.show()
