'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse
import copy
from keras.layers import Input
from imageio import imsave
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from utils import *
from adapt import *
import os
import time
import math
# read the parameter
# argument parsing

parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('model_selection', help="Choice of neural network model", type=int, choices=[1, 2, 3])
args = parser.parse_args()
# input image dimensions
img_rows, img_cols = 224, 224
# top_k for specifying the number of most active neurons
top_k = 3
# Number of selected neurons
k = 10
# Enter the size of the picture
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# input address
img_paths = "./seeds_10"
# load multiple models sharing same input tensor
if args.model_selection == 1:
    model1 = VGG16(input_tensor=input_tensor)
elif args.model_selection == 2:
    model1 = VGG19(input_tensor=input_tensor)
else:
    model1 = ResNet50(input_tensor=input_tensor)
layer_names = [layer.name for layer in model1.layers if
               'flatten' not in layer.name and 'input' not in layer.name]
# init coverage table
orig_covered = init_table(model1)
all_covered = np.zeros_like(orig_covered, dtype="bool")
# init table of the number of neuron coverage
covered_count = init_covered_count(orig_covered)
# init table of adversarial sample neuron coverage
objective_covered = init_objective_covered(orig_covered)
# ==============================================================================================
# start gen inputs
# Number of adversarial samples
adv_number = 0
# Clean up the memory every certain amount
i_number_clear = 0
adv_number_list = [[0] * 1000] * 1000
adv_list_number = 0
coverage_list = []
# init the strategy
strategies = init_strategies()
strategy = strategies.pop(0)
# storage strategy and strategy coverage information
records = []
diff_all = 0
star_time = time.time()
pre_covered = 0
while True:
    # Number of original seeds
    i_number = 0
    for filename in os.listdir(img_paths):
        # Extract the original label of the input
        img_name = filename.split('.')[0]
        mannual_label = int(img_name.split('_')[1])

        # Get input
        img_path = './seeds_10/' + filename

        # Preprocess the input
        orig_img = preprocess_image(img_path)
        covered = create_covered(orig_img, model1, top_k)
        # Update coverage information
        orig_covered = np.bitwise_or(orig_covered, covered)
        all_covered = np.bitwise_or(all_covered, covered)
        # Create seed set
        img_list = []
        temp_img = orig_img.copy()
        img_list.append(temp_img)
        next_covered = copy.deepcopy(all_covered)
        i_number += 1
        # achieve the const_vectors
        const_vectors = Creat_Features(model1, orig_img)
        # init table of strategy
        strategy_covered = np.zeros_like(orig_covered, dtype=bool)
        # Iterate over the seed set
        while len(img_list) > 0:
            # Clean up the memory when a certain number is reached
            if i_number_clear % 10 == 0:
                K.clear_session()
                input_tensor = Input(shape=input_shape)
                K.set_learning_phase(0)
                if args.model_selection == 1:
                    model1 = VGG16(input_tensor=input_tensor)
                elif args.model_selection == 2:
                    model1 = VGG19(input_tensor=input_tensor)
                else:
                    model1 = ResNet50(input_tensor=input_tensor)
                print(' pre covered neurons %.4f'
                      % (pre_covered))

                print(i_number)
                print('covered neurons percentage %d neurons %.4f'
                      % (len(orig_covered), np.mean(orig_covered)))
                print('attacks covered neurons percentage %d neurons %.4f'
                      % (len(all_covered), np.mean(all_covered)))
            i_number_clear += 1
            # Get the first input in the seed set and remove it
            gen_img = img_list[0]
            img_list.remove(gen_img)

            # Get the input label
            pred1 = model1.predict(gen_img)
            label1 = np.argmax(pred1[0])
            if label1 != mannual_label:
                continue
            orig_label = label1
            # select the neuron
            neurons = select(k, const_vectors, covered_count, objective_covered, strategy, model1, gen_img)
            if args.model_selection == 3:
                loss1 = -args.weight_diff * K.mean(model1.get_layer('fc1000').output[..., orig_label])
            else:
                loss1 = -args.weight_diff * K.mean(model1.get_layer('predictions').output[..., orig_label])
            # construct joint loss function
            for li, ni in neurons:
                loss1_neuron = K.sum(K.mean(model1.get_layer(layer_names[li]).output[..., ni]))
            layer_output = loss1 + 0.5 * loss1_neuron

            # for adversarial image generation
            final_loss = K.mean(layer_output)
            # we compute the gradient of the input picture wrt this loss
            grads = normalize(K.gradients(final_loss, input_tensor)[0])

            # this function returns the loss and grads given the input picture
            iterate = K.function([input_tensor], [loss1, loss1_neuron, grads])
            # we run gradient ascent for 5 steps

            for iters in range(args.grad_iterations):
                loss_value1, loss_neuron1, grads_value = iterate([gen_img])
                gen_img += grads_value * args.step
                # achieve the input coverage
                covered = create_covered(gen_img, model1, top_k)
                # update the strategy coverage
                strategy_covered = np.bitwise_or(strategy_covered, covered)
                # update the table of the number of neuron coverage
                covered_count += covered
                predictions1 = np.argmax(model1.predict(gen_img)[0])
                if predictions1 != label1:
                    all_covered = np.bitwise_or(all_covered, covered)
                    # update the table of adversarial sample neuron coverage
                    objective_covered = np.bitwise_or(objective_covered, covered)
                    gen_temp_img = gen_img.copy()
                    gen_img_deprocessed = deprocess_image(gen_temp_img)
                    # Save adversarial input
                    imsave('./res_4/' + str(adv_number) + '_' + str(iters) + '__' +
                           str(label1) + '__' + str(predictions1) + '_' + '.png', gen_img_deprocessed)
                    diff_img_adv = gen_temp_img - orig_img
                    l2_adv_norm = np.linalg.norm(diff_img_adv)
                    diff_all += l2_adv_norm
                    adv_number += 1
                    if (adv_number_list[label1][predictions1] == 0):
                        adv_number_list[label1][predictions1] = 1
                        adv_list_number += 1
                    break
            pre_coverage = np.mean(next_covered)
            next_covered = np.bitwise_or(next_covered, covered)
            diff_img = gen_img - orig_img
            l2_norm = np.linalg.norm(diff_img)
            orig_l2_norm = np.linalg.norm(orig_img)
            perturb_adversial = l2_norm / orig_l2_norm
            if np.mean(next_covered) - pre_coverage > 0.000001 and perturb_adversial < 0.05:
                gen_temp_img2 = gen_img.copy()
                img_list.append(gen_temp_img2)
        strategy = next(records, strategy, strategy_covered, strategies)
        now_coverage = np.mean(all_covered)
        coverage_list.append(now_coverage)
    end_time = time.time()
    all_time = end_time - star_time
    now_coverage = np.mean(all_covered)
    if all_time > 43200 and now_coverage - pre_covered < 0.01:
        print("on")
        break
    pre_covered = now_coverage
end_time = time.time()
cost_time = end_time - star_time
print('covered neurons percentage %d neurons %.4f'
      % (len(orig_covered), np.mean(orig_covered)))
print('attacks covered neurons percentage %d neurons %.4f'
      % (len(all_covered), np.mean(all_covered)))
print("adv number:" + str(adv_number))
print("adv label number:" + str(adv_list_number))
print("dis:" + str(diff_all / adv_number))
print(cost_time)
