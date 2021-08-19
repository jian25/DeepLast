import random
from collections import defaultdict, OrderedDict
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions


# util function to convert a tensor into a valid image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    # input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
    return input_img_data




def deprocess_image(x):
    x = x.reshape((224, 224, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.799
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def decode_label(pred):
    return decode_predictions(pred)[0][0][1]


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def init_table(model):
    covered_list = []
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        covered_layer_list = np.zeros(layer.output_shape[-1], dtype="bool")
        covered_list.append(covered_layer_list)
    return np.concatenate(covered_list)


def neuron_covered(covered):
    covered_neurons = float(np.mean(covered))
    total_neurons = len(covered)
    return total_neurons, covered_neurons


def create_covered(input_data, model, k):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    covered = []
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        layer_output_dict = {}
        layer_output = intermediate_layer_output[0]
        covered_layer = np.zeros(layer_output.shape[-1], dtype="bool")
        for num_neuron in range(layer_output.shape[-1]):
            output = np.mean(layer_output[..., num_neuron])
            layer_output_dict[num_neuron] = output
        sorted_index_output_dict = OrderedDict(
            sorted(layer_output_dict.items(), key=lambda x: x[1], reverse=True))
        dict_num = 1
        for key in sorted_index_output_dict.keys():
            if dict_num == k + 1:
                break
            covered_layer[key] = True
            dict_num += 1
        covered.append(covered_layer)
    return np.concatenate(covered)


def init_covered_count(covered):
    covered_count = np.zeros_like(covered, dtype="int")
    return covered_count


def init_objective_covered(covered):
    objective_covered = np.zeros_like(covered, dtype="bool")
    return objective_covered


