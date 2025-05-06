# Geology Forecast
Sequence to sequence models (encoder-decoder LSTMs) to predict geological profiles from initial sequence.
Challenge and data posted on [Kaggle]{https://www.kaggle.com/competitions/geology-forecast-challenge-open}.

### data:
- train_raw: directory, which includes raw geological profiles used for training
- test.csv: includes initial sequence of geological profiles used for test prediction
- sample_submission.csv: example format to submit model prediction to challenge

### code:
- interpolate_and_split.py: original code made available form challenge. Modified to create explanatory figures and output statistics, include data augmentation (rotate each profile by 180 degree) and changed sampling of input data to expand the training data set (stride < full sequence length, sequence length randomly chosen from distribution of sequence lengths in test data)
- encoder_decoder_lstm.py: architecture, backpropagation and training of encoder-decoder LSTM
- implementation.py: script for grid search
- plotting.py: script to create figures of exemplary predictions
- test_prediction.py: script to load model and create csv output for test prediction
- evaluation_metric.py: code provided by contest authors. Modified to plot weighting of prediction error over horizontal distance
