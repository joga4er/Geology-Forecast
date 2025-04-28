import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
import os

from encoder_decoder_lstm import lstm_seq2seq
from plotting import plot_train_test_results
from test_prediction import make_test_prediction

# parameters
len_training = 0.9
len_validation = 1
forecast_sequence_length = 300
input_sequence_length = 300

# read in training data
training_validation = pd.read_csv(r"data\train_augmented.csv", index_col=[0])

len_training = int(len_training * len(training_validation))
len_validation = int(len_validation * len(training_validation))

x_train = training_validation.iloc[:len_training, :(input_sequence_length - 1)].to_numpy().transpose()
x_train = pad_sequence([torch.from_numpy(np.array([i for i in x_row if pd.notnull(i)])) for x_row in x_train]).T
x_train = x_train[:, :, None].type(torch.Tensor)

y_train = training_validation.iloc[:len_training, input_sequence_length:].to_numpy().transpose()
y_train = torch.from_numpy(y_train[:, :, None]).type(torch.Tensor)

x_validation = training_validation.iloc[len_training:len_validation, :(input_sequence_length - 1)].to_numpy().transpose()
x_validation = pad_sequence([torch.from_numpy(np.array([i for i in x_row if pd.notnull(i)])) for x_row in x_validation]).T
x_validation = x_validation[:, :, None].type(torch.Tensor)

y_validation = training_validation.iloc[len_training:len_validation, input_sequence_length:].to_numpy().transpose()
y_validation = torch.from_numpy(y_validation[:, :, None]).type(torch.Tensor)

grid_search = pd.DataFrame({
    "hidden_size":[], "num_layers":[], "batch_size":[],
    "epochs":[], "train_loss":[], "val_loss":[], 
})
index = 0

for hidden_size in [4, 8, 12, 16, 32]:
    for num_layers in [1, 2, 3]:
        for batch_size in [50, 100, 200]:
            step_identifier = f'layers_{num_layers}_weights_{hidden_size}_batchsize_{batch_size}'

            # initialize model
            model = lstm_seq2seq(input_size=1, hidden_size=8, num_layers=2)
            # specify model parameters and train
            train_losses, val_losses, epochs = model.train_model(
                x_train, y_train, n_epochs=10, target_len=forecast_sequence_length, batch_size=100,
                validation_input_tensor=x_validation, validation_target_tensor=y_validation,
                description=step_identifier, training_prediction='mixed_teacher_forcing', teacher_forcing_ratio=1.0, 
                learning_rate=1e-3, dynamic_tf = True
                )
            print(f"{step_identifier}, {epochs}, {train_losses[-2]}, {val_losses[-2]}")
            grid_search.loc[index,:] = [
                hidden_size, num_layers, batch_size, epochs, train_losses[-2], val_losses[-2]
            ]

            # plot predictions on train/test data
            plot_train_test_results(
                model, x_train, y_train, x_validation, y_validation, description=step_identifier
                )
            make_test_prediction(model)
grid_search.to_csv(os.path.join('data', 'grid_search.csv'))

#!TODO
# implement everything with subset of data (faster)!
# early stoppings: save previous model.
# ensure test prediction works.
# parameter studies:
# batch size, number of layers, length of hidden state, learning rate?, teacher_forcing?
# skip zeros as end of sequence? (very bad input for decoder lstm)
# dropout, bidirectional
# try my luck with probibilistic (baysian) RNN as decoder