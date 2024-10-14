from keras import layers
#from trans_model import TransactionLayer, BinarizeLayer



class PatternMatch(layers.Layer):
    def __init__(self, trans_model, **kwargs):
        super().__init__(**kwargs)
        self.trans_model = trans_model

    def compute_output_shape(self, input_shape):
        return [None, 1]

    def call(self, inputs):
        outs = self.trans_model(inputs)
        print(1)
