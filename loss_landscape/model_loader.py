import os
import cifar10.model_loader

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    return net

def my_load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar100':
        net = cifar10.model_loader.my_load(model_name, model_file, data_parallel)
    return net
