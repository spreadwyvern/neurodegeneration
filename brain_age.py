import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import test_model

import os
import numpy as np

import argparse
# import pandas as pd
# from pandas import Series

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_id', type=str, default="0", help='GPU ID')
    parser.add_argument('-d', '--dir', type=str, default='model_for_prediction/model.pkl', help='path of trained model weight')

    FLAG = parser.parse_args()

    brain_age(FLAG)

def change_format(img):
        img = np.expand_dims(img, axis = 3)
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 0, 3)
        return img


def brain_age(FLAG):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAG.gpu_id

    torch.cuda.set_device(0)
    # specify dtype
    use_cuda = torch.cuda.is_available()
    print("Predict with GPU: {}".format(use_cuda))
    if use_cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    
    dir_m = FLAG.dir

    # build model and load trained weights
    model = test_model.CNN_2(base=32).type(dtype)
    net = torch.nn.DataParallel(model)

    net.load_state_dict(torch.load(dir_m))


    # begin prediction
    while True:
        # load MD numpy array
        MD_path = input('Input MD path: ')
        print(MD_path)
        try:
            X_MD = np.load(MD_path)
        except:
            print('Unable to open MD images! Try again!')
            continue
        X_MD = change_format(X_MD)
        X_MD = np.expand_dims(X_MD, axis = 0)
        X_MD = torch.from_numpy(X_MD)
        
        # load FA numpy array
        FA_path = input('Input FA path: ')
        try:
            X_FA = np.load(FA_path)
        except:
            print('Unable to open FA images! Try again!')
            continue
        X_FA = change_format(X_FA)
        X_FA = np.expand_dims(X_FA, axis = 0)
        X_FA = torch.from_numpy(X_FA)

        # X_MD = Variable(X_MD, volatile=True).type(dtype)
        # X_FA = Variable(X_FA, volatile=True).type(dtype)
        X_MD = X_MD.type(dtype)
        X_FA = X_FA.type(dtype)

        # prediction
        with torch.no_grad():
            toutput = net(X_MD, X_FA)
        
        pred = toutput.data.cpu().numpy()
        
        print('Predicted brain age: {} years old'.format(pred))
        
        del X_FA
        del X_MD

if __name__ == '__main__':
    main()
