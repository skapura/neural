from keras import models


def make_output_model(model, layers):
    outputs = [layer.output for layer in model.layers if layer.name in layers]
    outputmodel = models.Model(inputs=model.inputs, outputs=outputs)
    return outputmodel
