# Geology Forecast
Probabilistic sequence to sequence models (LSTMs) to predict geological profiles from initial sequence.
Challenge and data posted on [Kaggle]{https://www.kaggle.com/competitions/geology-forecast-challenge-open}.
Draft version, models not implemented yet.

### data:
- train_raw: directory, which includes raw geological profiles used for training
- test.csv: includes initial sequence of geological profiles used for test prediction
- sample_submission.csv: example format to submit model prediction to challenge

### code:
- interpolate_and_split.py: original code made available form challenge. Modified to create explanatory figures and output statistics, include data augmentation (rotate each profile by 180 degree) and changed sampling of input data to expand the training data set (stride < full sequence length, sequence length randomly chosen from distribution of sequence lengths in test data)# Geology Forecast
 Source code for the prediction of geological profiles
 - evaluation_metric.py: code provided by contest authors. Modified in order to plot standard deviation of profile over horizontal distance
