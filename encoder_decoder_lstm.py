# reference:
# https://github.com/lkulowski/LSTM_encoder_decoder/blob/master/code/lstm_encoder_decoder.py

# Author: Laura Kulowski

import numpy as np
import random
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import os

class probabilistic_linear_layer(nn.Module):
    """
    source: https://medium.com/@pumplerod/probabilistic-neural-network-with-pytorch-11ec04479f67
    """
    def __init__(
            self, in_features, out_features, bias=True, 
            distribution='uniform', init_type='fan_in',
            device=None, dtype=torch.float32,**kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        #
        # epsilon added to 'var' so that scale is always > 0 or that high > low
        self.eps = torch.tensor( [1e-6])
        self.distribution = distribution
        self.in_features = in_features
        self.out_features = out_features
        assert self.distribution.upper() in ['UNIFORM', 'NORMAL'], f"Distribution must be one of: ['uniform', 'normal']. Not: {self.distribution}"
        
        if init_type.upper()=='FAN_OUT':
            kaiming_scale = torch.Tensor([np.sqrt(6 / out_features)]).type( torch.float32)
        else:
            kaiming_scale = torch.Tensor([np.sqrt(6 / in_features)]).type( torch.float32)
                                        
        self.w_val = torch.nn.parameter.Parameter( torch.randn( (out_features, in_features), **factory_kwargs) * kaiming_scale, requires_grad=True)
        self.w_var = torch.nn.parameter.Parameter( torch.ones_like( self.w_val, **factory_kwargs) * kaiming_scale, requires_grad=True)
        
        if bias:
            self.b_val = torch.nn.parameter.Parameter( torch.randn( (out_features), **factory_kwargs) * kaiming_scale, requires_grad=True)
            self.b_var = torch.nn.parameter.Parameter( torch.ones_like( self.b_val, **factory_kwargs) * kaiming_scale, requires_grad=True)
        else:
            self.b_val = self.register_parameter( 'b_val', None)
            self.b_var = self.register_parameter( 'b_var', None)

    def forward(self, x: torch.Tensor):
        #
        # We draw a new set of weight/bias each time the forward call is made.
        #   - I am using torch.abs() and a small epsilon in order to insure
        #     scale is always positive or that high > low.  There may be a
        #     much better way to go about this.
        if self.distribution.upper() == 'UNIFORM':
            weight = torch.distributions.Uniform( low=self.w_val-torch.abs( self.w_var), high=self.w_val+torch.abs(self.w_var)+self.eps.to( x.device)).rsample()
            bias = torch.distributions.Uniform( low=self.b_val-torch.abs(self.b_var), high=self.b_val+torch.abs(self.b_var)+self.eps.to( x.device)).rsample() if self.b_val is not None else 0.0
        elif self.distribution.upper() == 'NORMAL':
            weight = torch.distributions.Normal( loc=self.w_val, scale=torch.abs(self.w_var)+self.eps.to( x.device)).rsample()
            bias = torch.distributions.Normal( loc=self.b_val, scale=torch.abs(self.b_var)+self.eps.to( x.device)).rsample() if self.b_val is not None else 0.0

        return x@weight.T + bias
    
    #
    # Provide a little more information when this instance is printed
    def extra_repr(self) -> str:
        bias = self.b_val is not None
        return ' in_features={}, out_features={}, bias={}, prob_distribution={}, device={}'.format(
            self.in_features, self.out_features, bias, self.distribution.upper(), self.device
        )

    #
    # So that we can determine which device this is set to
    @property
    def device(self):
        return next(self.parameters()).device

class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers = 1):
        
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=0.1,
            )

    def forward(self, x_input):
        
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''

        lstm_out, self.hidden = self.lstm(x_input)
        
        return lstm_out, self.hidden     
    
    def init_hidden(self, batch_size, device):
        
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size, num_layers = 1):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=0.1,
            )
        # self.linear = nn.Linear(hidden_size, input_size)
        self.linear = probabilistic_linear_layer(
            in_features=hidden_size, out_features=1, bias=True, distribution='normal'
        )           

    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''
        
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))     
        
        return output, self.hidden

class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    
    def __init__(self, input_size, hidden_size, num_layers):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = lstm_encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = lstm_decoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def train_model(
            self, input_tensor, target_tensor, device, n_epochs, target_len, 
            batch_size, validation_input_tensor = None, validation_target_tensor = None,
            description = '', training_prediction = 'recursive', teacher_forcing_ratio = 0.6,
            learning_rate = 1e-3, dynamic_tf = False):
        
        '''
        train lstm encoder-decoder
        
        : param input_tensor:              input data with shape (seq_len, # in batch, number features); PyTorch tensor    
        : param target_tensor:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param n_epochs:                  number of epochs 
        : param target_len:                number of values to predict 
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch
        '''
        
        # initialize array of losses 
        training_losses = []
        validation_losses = []

        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # calculate number of batch iterations
        n_batches = int(input_tensor.shape[1] / batch_size)

        with trange(n_epochs) as tr:
            for it in tr:
                
                batch_loss = 0.

                self.train()
                for b in range(n_batches):
                    # select data 
                    input_batch = input_tensor[:, b: b + batch_size, :].to(device)
                    target_batch = target_tensor[:, b: b + batch_size, :].to(device)

                    # outputs tensor
                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])

                    # initialize hidden state
                    encoder_hidden = self.encoder.init_hidden(batch_size, device=device)

                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    _, encoder_hidden = self.encoder(input_batch)

                    # decoder with teacher forcing
                    decoder_input = input_batch[-1, :, :]   # shape: (batch_size, input_size)
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        # predict recursively
                        for t in range(target_len): 
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == 'teacher_forcing':
                        # use teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = target_batch[t, :, :]

                        # predict recursively 
                        else:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == 'mixed_teacher_forcing':
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            
                            # predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = target_batch[t, :, :]
                            
                            # predict recursively 
                            else:
                                decoder_input = decoder_output

                    # compute the loss
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()
                    
                    # backpropagation
                    loss.backward()
                    optimizer.step()

                # loss for epoch 
                batch_loss /= n_batches 
                training_losses.append(batch_loss)

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.1
                    if teacher_forcing_ratio < 0:
                        teacher_forcing_ratio = 0

                # progress bar 
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))

                # keep track of validation error, especially relevant for early stoppings
                validation_loss = 0.0
                if validation_input_tensor.nelement() > 0 \
                        and validation_target_tensor.nelement() > 0:
                    for forecast_index in range(validation_input_tensor.shape[1]):
                        prediction = self.predict(
                            validation_input_tensor[:,forecast_index,:], target_len=validation_target_tensor.shape[0]
                            )
                        prediction_loss = criterion(prediction, validation_target_tensor[:,forecast_index,:])
                        validation_loss += prediction_loss.item()
                    validation_losses.append(validation_loss/forecast_index)
                    if it == 4:
                        torch.save(
                            self.state_dict(), os.path.join('model', description)
                            )
                    if it > 4: # early stoppings after 5 epochs
                        if validation_losses[it] >= validation_losses[it - 1]:
                            self.load_state_dict(torch.load(
                                os.path.join('model', description), weights_only=True)
                                )
                            break
                        else:
                            torch.save(
                                self.state_dict(), os.path.join('model', description)
                            )
                else:
                    validation_losses.append(np.nan)
        plotname = f'losses_{description}.png'
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(training_losses)), training_losses, label='training')
        ax.plot(np.arange(len(validation_losses)), validation_losses, label='validation')
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.savefig(os.path.join('plots', plotname))
        plt.close() 
                    
        return training_losses, validation_losses, it
    
    def predict(self, input_tensor, target_len):
        
        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        '''
        # change to evaluation mode -> relavant to not do dropout
        self.eval()

        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(1)     # add in batch size of 1
        _, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2])

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden
        
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output
        
        return outputs