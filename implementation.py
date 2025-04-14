import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from encoder_decoder_lstm import lstm_seq2seq
from plotting import plot_train_test_results

# read in training data
training_validation = pd.read_csv(r"data\train_augmented.csv", index_col=[0])

x_train = training_validation.iloc[:int(0.75*len(training_validation)), :300].to_numpy()
x_train = pad_sequence([torch.from_numpy(np.array([i for i in x_row if pd.notnull(i)])) for x_row in x_train])
y_train = training_validation.iloc[:int(0.75*len(training_validation)):, 300:].to_numpy().transpose()

x_validation = training_validation.iloc[int(0.75*len(training_validation)):, :300].to_numpy()
x_validation = pad_sequence([torch.from_numpy(np.array([i for i in x_row if pd.notnull(i)])) for x_row in x_validation])
y_validation = training_validation.iloc[int(0.75*len(training_validation)):, 300:].to_numpy().transpose()

# only one feature, still add third dimension
X_train_torch = x_train[:, :, None].type(torch.Tensor)
Y_train_torch = torch.from_numpy(y_train[:, :, None]).type(torch.Tensor)

X_validation_torch = x_validation[:, :, None].type(torch.Tensor)
Y_validation_torch = torch.from_numpy(y_validation[:, :, None]).type(torch.Tensor)

print(X_train_torch.shape)
print(Y_train_torch.shape)

# specify model parameters and train
model = lstm_seq2seq(input_size=X_train_torch.shape[2], hidden_size=15)
loss = model.train_model(X_train_torch, Y_train_torch, n_epochs=50, target_len=300, batch_size=100, training_prediction='mixed_teacher_forcing', teacher_forcing_ratio=0.6, learning_rate=0.01, dynamic_tf = False)

# plot predictions on train/test data
plot_train_test_results(model, X_train_torch, Y_train_torch, X_validation_torch, Y_validation_torch)

plt.close('all')

#!TODO
# make test prediction (copy 10 times)
# include loss function from contest
# try my luck with probibilistic (baysian) RNN as decoder