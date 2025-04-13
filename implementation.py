import pandas as pd
import numpy as np
import torch

# read in training data
training_validation = pd.read_csv(r"data\train_augmented.csv", index_col=[0])

x_train = training_validation.iloc[:int(0.75*len(training_validation)), :300].to_numpy()
y_train = training_validation.iloc[int(0.75*len(training_validation)):, 300:].to_numpy()
x_validation = training_validation.iloc[:int(0.75*len(training_validation)), 300:].to_numpy()
y_validation = training_validation.iloc[int(0.75*len(training_validation)):, 300:].to_numpy()

X_train_torch = torch.from_numpy(x_train).type(torch.Tensor)
Y_train_torch = torch.from_numpy(y_train).type(torch.Tensor)

X_validation_torch = torch.from_numpy(x_validation).type(torch.Tensor)
Y_validation_torch = torch.from_numpy(y_validation).type(torch.Tensor)

#!TODO
# feed into model
# make encoder LSTM skip NaNs at beginning of sequence
# make test prediction (copy 10 times)
# include loss function from contest
# try my luck with probibilistic (baysian) RNN as decoder