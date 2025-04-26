import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence

from encoder_decoder_lstm import lstm_seq2seq
from plotting import plot_train_test_results
from test_prediction import make_test_prediction

# read in training data
training_validation = pd.read_csv(r"data\train_augmented.csv", index_col=[0])

x_train = training_validation.iloc[:int(0.75*len(training_validation)), :300].to_numpy().transpose()
x_train = pad_sequence([torch.from_numpy(np.array([i for i in x_row if pd.notnull(i)])) for x_row in x_train]).T
x_train = x_train[:, :, None].type(torch.Tensor)

y_train = training_validation.iloc[:int(0.75*len(training_validation)):, 300:].to_numpy().transpose()
y_train = torch.from_numpy(y_train[:, :, None]).type(torch.Tensor)

x_validation = training_validation.iloc[int(0.75*len(training_validation)):, :300].to_numpy().transpose()
x_validation = pad_sequence([torch.from_numpy(np.array([i for i in x_row if pd.notnull(i)])) for x_row in x_validation]).T
x_validation = x_validation[:, :, None].type(torch.Tensor)

y_validation = training_validation.iloc[int(0.75*len(training_validation)):, 300:].to_numpy().transpose()
y_validation = torch.from_numpy(y_validation[:, :, None]).type(torch.Tensor)


# specify model parameters and train
model = lstm_seq2seq(input_size=1, hidden_size=8)
loss = model.train_model(
    x_train, y_train, n_epochs=10, target_len=300, batch_size=100,
    validation_input_tensor=x_validation, validation_target_tensor=y_validation,
    training_prediction='mixed_teacher_forcing', teacher_forcing_ratio=0.6, 
    learning_rate=0.001, dynamic_tf = True
    )

# plot predictions on train/test data
plot_train_test_results(model, x_train, y_train, x_validation, y_validation)
make_test_prediction(model)

#!TODO
# early stoppings
# make test prediction (copy 10 times)
# parameter studies:
# batch size, number of layers, length of hidden state, learning rate?, teacher_forcing?
# try my luck with probibilistic (baysian) RNN as decoder