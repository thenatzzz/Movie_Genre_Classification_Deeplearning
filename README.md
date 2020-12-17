# Deeplearning for Movie Genre Classification based on Posters and Overviews
#### CMPT-726 Machine Learning (Simon Fraser University)
#### Group members: Nattapat Juthaprachakul, Rui Wang, Siyu Wu, Yihan Lan

### Abstract
The film industry increasing relies upon movie posters and overviews in order to attract public attention in the hopes of generating huge profit and viewership. A genre of movie is one of the very important factors that helps movie goers decide which movies they want to spend time and money on./

The project uses a poster and overview of movie to predict its genre. Therefore, our goal is to build and compare several multilabel-multiclass classification models based on classical machine learning algorithm and deeplearning-based models./

 CNN model is used to train our model with an image feature while Random Forest and LSTM models are applied to train ours based on a text feature. Later, the combining models with the text and image feature are used to obtain comprehensive results./
 In this project, we found that the combination of LSTM and custom CNN model is reasonably successful at predicting genres with the highest at-least-one-matched accuracy is at 65.46%.

### Methods
* Convolutional Neural Networks on poster datasets
* Random Forest on movie overviews
* LSTM on movie overviews
* Combination of models on both poster and overviews

### Dataset
* MovieLens
* TMDB

### Evaluation
* F1, Precision, Recall
* Hamming Loss
* At least one match, All matches

### Program Requirement
* Tensorflow, Keras
* Python 3+
* Python libraries: Pandas, Numpy, Matplotlib, etc.
