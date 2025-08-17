import os
import time
import math
import json
import copy as cp
import random as rd
import pickle
import gzip
import yaml
import logging
import threading
from io import BytesIO
from PIL import Image, ImageFilter
from collections import OrderedDict, Counter, defaultdict, deque

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST

import flask
from flask import Flask, request, send_from_directory, jsonify, abort
from werkzeug.utils import secure_filename

import aiohttp
import aiofiles
import asyncio

# Load configuration
with open('./Config.yaml', 'r') as file:
    config = yaml.safe_load(file)

seed = config['dataset']['random_seed']
np.random.seed(seed)
rd.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

def compress_model_untarget(gradients, p):
    compressed_gradients = {}
    for key, value in gradients.items():
        tensor = value.flatten()
        num_elements = tensor.numel()
        num_selected = int(num_elements * p)

        _, indices = torch.topk(tensor.abs(), num_selected, largest=True, sorted=False)
        compressed_tensor = tensor[indices].to(torch.float16)
        compressed_indices = indices.to(torch.int32)

        compressed_gradients[key] = (compressed_tensor, compressed_indices)

    return pickle.dumps(compressed_gradients)

def compress_model_target(gradients, goal_gradients):
    compressed_gradients = {}
    for key, value in gradients.items():
        tensor = value.flatten()
        goal_tensor = goal_gradients[key].flatten()

        indices = torch.nonzero(goal_tensor, as_tuple=True)[0]

        compressed_tensor = tensor[indices].to(torch.float16)
        compressed_indices = indices.to(torch.int32)

        compressed_gradients[key] = (compressed_tensor, compressed_indices)

    return pickle.dumps(compressed_gradients)

def decompress_model(compressed_gradients, original_shape_dict):
    compressed_gradients = pickle.loads(compressed_gradients)
    decompressed_gradients = {}
    for key, (compressed_tensor, compressed_indices) in compressed_gradients.items():
        original_shape = original_shape_dict[key]
        decompressed_tensor = torch.zeros(original_shape.numel(), dtype=torch.float32)
        decompressed_tensor[compressed_indices.long()] = compressed_tensor.to(torch.float32)
        decompressed_gradients[key] = decompressed_tensor.view(original_shape)
    return decompressed_gradients

def save_dataloader_info(dataset_info, data_loader_train, data_loader_test, filename):
    info = {
        'dataset_info': dataset_info,
        'dataloader_train': data_loader_train,
        'dataloader_test': data_loader_test
    }
    with open(filename, 'wb') as f:
        pickle.dump(info, f)

def load_dataloader_info(filename):
    with open(filename, 'rb') as f:
        info = pickle.load(f)
    dataset_info = info['dataset_info']
    data_loader_train = info['dataloader_train']
    data_loader_test = info['dataloader_test']
    return dataset_info, data_loader_train, data_loader_test

def save_compressed_state_dict(compressed_state_dict, filepath):
    with gzip.open(filepath, 'wb') as f:
        f.write(compressed_state_dict)

def load_compressed_state_dict(filepath):
    with gzip.open(filepath, 'rb') as f:
        compressed_state_dict = f.read()
    return compressed_state_dict

def calculate_state_dict_size(state_dict):
    total_size = 0
    for key, value in state_dict.items():
        total_size += value.numel() * value.element_size()
    return total_size

def create_selected_state_dict(original_state_dict, goal_state_dict):
    selected_state_dict = {}
    for key in goal_state_dict:
        indices = (torch.abs(goal_state_dict[key]) > 0)
        selected_state_dict[key] = original_state_dict[key] * indices
    return selected_state_dict







