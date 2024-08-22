from keras import models


class PatternModel(models.Model):

    def __init__(self, layers, **kwargs):
        super().__init__(**kwargs)
        self.receptive_info = conv_layer_info(self, self.output[-2].name)
        self.output_names = layers
        for i in range(len(layers)):
            self.output[i].name = layers[i]

    def receptive_subset(self, layer):
        idx = self.layers.index(self.get_layer(layer)) - 1
        name = self.layers[idx].name
        rlayers = []
        for l in self.receptive_info:
            rlayers.append(l)
            if l['name'] == name:
                break
        return rlayers

    def filters_in_layer(self, layer):
        numfilters = self.get_layer(layer).output.shape[-1]
        return [layer + '-' + str(i) for i in range(numfilters)]


def make_output_model(model, layers):
    outputs = [layer.output for layer in model.layers if layer.name in layers]
    #outputmodel = models.Model(inputs=model.inputs, outputs=outputs)
    outputmodel = PatternModel(layers, inputs=model.inputs, outputs=outputs)
    return outputmodel


def conv_layer_info(model, last=None):
    convinfo = []
    for i in range(len(model.layers)):
        if last is not None and model.layers[i].name == last:
            break
        l = model.layers[i]
        if 'conv2d' in l.name or 'max_pooling2d' in l.name or l.name.startswith('average_pooling2d'):
            convinfo.append({'name': l.name, 'kernel': l.kernel_size if 'conv2d' in l.name else l.pool_size, 'stride': l.strides})

    return convinfo


def layer_subset(layers, last):
    selected = []
    for l in layers:
        selected.append(l)
        if l == last:
            break
    selected.append(layers[-1])
    return selected


def parse_feature_ref(name):
    tokens = name.split('-')
    return tokens[0], int(tokens[1])


def receptive_field(x, y, layers):
    startx = x
    endx = x
    starty = y
    endy = y
    for l in reversed(layers):
        startx = startx * l['stride'][1]
        endx = endx * l['stride'][1] + l['kernel'][1] - 1
        starty = starty * l['stride'][0]
        endy = endy * l['stride'][0] + l['kernel'][0] - 1
    return (startx, endx), (starty, endy)
