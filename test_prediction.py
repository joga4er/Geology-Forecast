from encoder_decoder_lstm import lstm_seq2seq

import torch
from torch.nn.utils.rnn import pad_sequence

import os
import pandas as pd
import numpy as np

def make_test_prediction(model: lstm_seq2seq):
    test_data = pd.read_csv(os.path.join('data','test.csv'), index_col=[0])
    result_data = pd.read_csv(os.path.join('data','sample_submission.csv'), index_col=[0])

    for (forecast_index, forecast_input) in test_data.iterrows():
        x_train = forecast_input.dropna().to_numpy()[:-1]
        x_train_tensor = torch.zeros(len(x_train), 1)
        x_train_tensor[:, 0] = torch.from_numpy(x_train)
        for model_index in range(10):
            forecast = model.predict(x_train_tensor, target_len=300)
            forecast = forecast.detach().numpy()
            result_data.loc[forecast_index, result_data.columns[
                model_index * 300: (model_index + 1) * 300]] = forecast[:,0]

    result_data.to_csv(os.path.join('data', 'submission.csv'), index=True)