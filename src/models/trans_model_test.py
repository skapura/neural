import tensorflow as tf
import keras
from keras import models
from keras.src.models import Model
import trans_model as tm
from src.models.trans_model import TransactionLayer, build_feature_extraction
import src.models.trans_model as tm
import src.data as data
import src.mlutil as mlutil


def TransactionLayer_test():
    #tf.config.run_functions_eagerly(True)
    b = tf.constant([
        [[1.11, 2.11, 3.11, 4.11], [1.21, 2.21, 3.21, 4.21], [1.31, 2.31, 3.31, 4.31]],
        [[1.12, 2.12, 3.12, 4.12], [1.22, 2.22, 3.22, 4.22], [1.32, 2.32, 3.32, 4.32]],
        [[1.13, 2.13, 3.13, 4.13], [1.23, 2.23, 3.23, 4.23], [1.33, 2.33, 3.33, 4.33]]
    ])
    batch = tf.expand_dims(b, axis=0)
    t = TransactionLayer()
    res = t(batch)
    print(res)


def Binarize_test():
    trainds = data.load_from_directory('datasets/images_large/train', shuffle=True, sample_size=100)
    basemodel = models.load_model('session/base.keras')
    outputlayers = ['activation', 'activation_1']
    tmodel = tm.build_transaction_model(basemodel, trainds, outputlayers)
    trans = tmodel.predict(trainds)
    bds = tm.transactions_to_dataframe(tmodel, trans)
    print(bds)

def run():
    #TransactionLayer_test()
    Binarize_test()
    print(1)
