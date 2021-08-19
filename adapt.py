from math import ceil
from keras import backend as K
from keras.models import Model
import numpy as np

# Number of features.
CONST_FEATURES = 17
VARIABLE_FEATURES = 12
TOTAL_FEATURES = 29
bound = 5
size = 100
history = 300
remainder = 0.5
sigma = 1


def Creat_Features(model, input_data):
    const_vectors = []
    # For f4-9.
    weights = []
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    # For each layer.
    for li, l in enumerate(intermediate_layer_outputs):

        # For f0-3.
        layer_location = li / (len(layer_names) - 1)
        layer_location = int(layer_location * 4)

        # For f4-9.
        w = model.get_layer(layer_names[li]).get_weights()
        if len(w) > 0:
            # Use only weights, not biases.
            w = w[0]
        else:
            # If layer without weights (e.g. Pooling layer).
            w = np.zeros(model.get_layer(layer_names[li]).output_shape[-1])

        # For f10-16.
        layer_type = \
            10 if "bn" in layer_names[li] else \
                11 if "pool" in layer_names[li] and "pool1" not in layer_names[li] else \
                    12 if "conv" in layer_names[li] or "pad" in layer_names[li] else \
                        13 if "fc" in layer_names[li] else \
                            14 if "activation" in layer_names[li] else \
                                15 if "add" in layer_names[li] else \
                                    16

        # For each neuron.
        layer_output = l[0]
        for ni in range(layer_output.shape[-1]):
            # For f4-9.
            weights.append(np.mean([w[..., ni]]))

            # Create a constant feature vector.
            vec_c = np.zeros(CONST_FEATURES, dtype=int)

            #  f0: Ff layer is located in front 25 percent.
            #  f1: Ff layer is located in between front 25 percent and front 50 percent.
            #  f2: Ff layer is located in between back 50 percent and back 25 percent.
            #  f3: Ff layer is located in back 25 percent.
            vec_c[layer_location] = 1

            # f10: If layer is normalization layer.
            # f11: If layer is pooling layer.
            # f12: If layer is convolution layer.
            # f13: If layer is dense layer.
            # f14: If layer is activation layer.
            # f15: If layer gets inputs from multiple layers.
            # f16: Otherwise.
            vec_c[layer_type] = 1

            const_vectors.append(vec_c)

    const_vectors = np.array(const_vectors)

    # For f4-f9.
    argsort_weights = np.argsort(weights)
    top_10 = int(len(argsort_weights) * 0.9)
    top_20 = int(len(argsort_weights) * 0.8)
    top_30 = int(len(argsort_weights) * 0.7)
    top_40 = int(len(argsort_weights) * 0.6)
    top_50 = int(len(argsort_weights) * 0.5)
    # f4: If neuron have weight of top 10%.
    const_vectors[argsort_weights[top_10:], 4] = 1
    # f5: If neuron have weight between top 10% and top 20%.
    const_vectors[argsort_weights[top_20:top_10], 5] = 1
    # f6: If neuron have weight between top 20% and top 30%.
    const_vectors[argsort_weights[top_30:top_20], 6] = 1
    # f7: If neuron have weight between top 30% and top 40%.
    const_vectors[argsort_weights[top_40:top_50], 7] = 1
    # f8: If neuron have weight between top 40% and top 50%.
    const_vectors[argsort_weights[top_50:top_40], 8] = 1
    # f9: If neuron have wiehgt of bottom 50%.
    const_vectors[argsort_weights[:top_50], 9] = 1
    return const_vectors


def updata_variable(const_vectors, objective_covered, covered_count):
    # Create variable feature vectors.
    variable_vectors = np.zeros((len(const_vectors), VARIABLE_FEATURES), dtype=int)

    # f17: If neuron activated when input satisfies objective function.
    indices = np.squeeze(np.argwhere(objective_covered > 0))
    variable_vectors[indices, 0] = 1

    # f18: If neuron never activated.
    indices = np.squeeze(np.argwhere(covered_count < 1))
    variable_vectors[indices, 1] = 1

    # For f19-28.
    sorted_indices = np.setdiff1d(np.argsort(covered_count), indices, assume_unique=True)
    top_10 = int(len(sorted_indices) * 0.9)
    top_20 = int(len(sorted_indices) * 0.8)
    top_30 = int(len(sorted_indices) * 0.7)
    top_40 = int(len(sorted_indices) * 0.6)
    top_50 = int(len(sorted_indices) * 0.5)
    top_60 = int(len(sorted_indices) * 0.4)
    top_70 = int(len(sorted_indices) * 0.3)
    top_80 = int(len(sorted_indices) * 0.2)
    top_90 = int(len(sorted_indices) * 0.1)
    # f19: If neuron activated top 10%.
    variable_vectors[sorted_indices[top_10:], 2] = 1
    # f20: If neuron activated between top 10% and top 20%.
    variable_vectors[sorted_indices[top_20:top_10], 3] = 1
    # f21: If neuron activated between top 20% and top 30%.
    variable_vectors[sorted_indices[top_30:top_20], 4] = 1
    # f22: If neuron activated between top 30% and top 40%.
    variable_vectors[sorted_indices[top_40:top_30], 5] = 1
    # f23: If neuron activated between top 40% and top 50%.
    variable_vectors[sorted_indices[top_50:top_40], 6] = 1
    # f24: If neuron activated between top 50% and top 60%.
    variable_vectors[sorted_indices[top_60:top_50], 7] = 1
    # f25: If neuron activated between top 60% and top 70%.
    variable_vectors[sorted_indices[top_70:top_60], 8] = 1
    # f26: If neuron activated between top 70% and top 80%.
    variable_vectors[sorted_indices[top_80:top_70], 9] = 1
    # f27: If neuron activated between top 80% and top 90%.
    variable_vectors[sorted_indices[top_90:top_80], 10] = 1
    # f28: If neuron activated between top 90% and top 100%.
    variable_vectors[sorted_indices[:top_90], 11] = 1
    return np.concatenate([const_vectors, variable_vectors], axis=1)


# select neurons
def select(k, const_vectors, covered_count, objective_covered, strategy, model, input_data):
    matrix = updata_variable(const_vectors, objective_covered, covered_count)
    scores = matrix.dot(strategy)
    neurons = []
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    for li, l in enumerate(intermediate_layer_outputs):
        layer_output = l[0]
        for ni in range(layer_output.shape[-1]):
            neurons.append((li, ni))

    # Get k highest neurons and return their location.
    indices = np.argpartition(scores, -k)[-k:]
    return [neurons[i] for i in indices]


def init_strategies():
    strategies = [np.random.uniform(-bound, bound, size=TOTAL_FEATURES) for _ in
                  range(size)]
    return strategies


def next(records, strategy1, strategy_covered, strategies):
    records.append((strategy1, strategy_covered))

    # Get the next strategy.
    if len(strategies) > 0:
        strategy = strategies.pop(0)
        strategy_covered = np.zeros_like(strategy_covered, dtype=bool)
        return strategy

    # Generate next strategies from the past records.
    records_now = records[-history:]

    # Find a set of strategies that maximizes the coverage.
    n = int(size * remainder)
    now_strategies, now_covereds = tuple(zip(*records_now))
    _, indices = greedy_max_set(now_covereds, n=n)

    # Find the maximum coverages for remaining part.
    n = n - len(indices)
    if n > 0:
        coverages = list(map(np.mean, now_covereds))
        indices = indices + list(np.argpartition(coverages, -n)[-n:])

    # Get strategies.
    selected = np.array(now_strategies)[indices]

    # Mix strategies randomly.
    n = len(selected)
    generation = ceil(1 / remainder)
    left = selected[np.random.permutation(n)]
    right = selected[np.random.permutation(n)]

    for l, r in zip(left, right):
        for _ in range(generation):
            # Generate new strategy.
            s = np.array(
                [l[i] if np.random.choice([True, False]) else r[i] for i in range(TOTAL_FEATURES)])

            # Add little distortion.
            s = s + np.random.normal(0, sigma, size=TOTAL_FEATURES)

            # Clip the ranges.
            s = np.clip(s, -bound, bound)

            strategies.append(s)

    strategies = strategies[:size]

    # Get the next strategy.
    strategy = strategies.pop(0)

    return strategy


def greedy_max_set(covereds, n=None):
    '''Returns a maximum coverage vector composed of n elements, and index of elements.

    This function solves maximum coverage problem with greedy approach.

    Args:
      covereds: A list of coverage vectors.
      n: Maximum number of elements to use for finding maximum coverage. By default,
        use the length of the `covereds`.
    '''

    # Check arguments.
    covereds = np.array(covereds)
    if n is None:
        n = len(covereds)

    # Initialize result variables.
    idxs = []
    max_set = np.zeros_like(covereds[0], dtype=bool)

    # Find n elements.
    for _ in range(n):

        # If there is no room for improvement.
        if np.sum(covereds) == 0:
            break

        # Find next greedy element.
        sums = [np.sum(m) for m in covereds]
        idx = np.argmax(sums)
        chosen = covereds[idx]

        # Update maximum coverage vector.
        max_set = np.bitwise_or(max_set, chosen)
        idxs.append(idx)

        # Update candidates.
        chosen = np.bitwise_not(chosen)
        covereds = [np.bitwise_and(m, chosen) for m in covereds]

    return max_set, idxs
