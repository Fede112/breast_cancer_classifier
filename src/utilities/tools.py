# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, 
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, 
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, 
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, 
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, 
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of breast_cancer_classifier.
#
# breast_cancer_classifier is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# breast_cancer_classifier is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with breast_cancer_classifier.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================
"""
Defines utility functions for various tasks in breast_cancer_classifier.
"""

import os
import numpy as np

def partition_batch(ls, size):
    """
    Partitions a list into buckets of given maximum length.
    """
    i = 0
    partitioned_lists = []
    while i < len(ls):
        partitioned_lists.append(ls[i: i+size])
        i += size
    return partitioned_lists


def get_activation(layer_dict, name):
    """
    Define hook to extract intermediate layer features
    """
    def hook(model, input, output):
        layer_dict[name].append(output.detach())
        # layer_dict[name] = torch.cat(layer_dict[name], output.detach())
    return hook


def save_activations(activations_dict, output_folder):
    """
    Save into separate files activations extracted with a hook. 
    One file per layer/activation.
    :param layer_dict: dictionary with the activations per layer.
    :param output_data_folder: path to folder where to store the activation files.
    """
    
    if os.path.exists(output_folder):
        # Prevent overwriting to an existing directory
        print("Error: the directory to save the activations already exists.")
        return
    else:
        os.makedirs(output_folder)


    for name, activation in activations_dict.items():
        file_path = os.path.join(output_folder,name + '.pkl')
        print(file_path)
        with open(file_path,'wb') as file: 
            np.save(file, activation.numpy())
            # np.savetxt(file, activation.numpy())


def load_activations(input_folder):
    """
    Load activations from different files into a single dictionary. 
    One file per layer/activation.
    :param layer_dict: dictionary with the activations per layer.
    :param output_data_folder: path to folder where to store the activation files.
    """
    activations = {}
    for file in os.listdir(input_folder):
        if file.endswith('.pkl'):
            full_path = os.path.join(input_folder, file)
            name = file[:-4]
            activation = load_single_activation(full_path)
            activations[name] = activation
    return activations


    
def load_single_activation(input_file):
    with open(input_file, 'rb') as file:
        return np.load(file)