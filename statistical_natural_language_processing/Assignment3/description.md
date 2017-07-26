### Pre-processing

* **Tokenisation:** We used our own regural expression for tokenisation. We experimented with the NTLK tokeniser but this did not yield improved performance. 

* **Word representations:**  We achieved best performance using pretrained GloVe word embeddings (with 200 dimensions), compared to randomly initialized embeddings, and GloVe 50d, 100d. We only use embeddings for words in the training vocabulary with new words being handled as OOV. 

### Initialisation 

* **Bucketing and batching:** Bucketing yields no improvement in performance but improves training times considerably. Random batches are drawn. The batch size is set to 1024, being tuned as a hyper-parameter.

* ** Truncated normal: ** We initialize the embedings weights with GloVe vectors. Words not contained in GloVe are initialized by truncated normal distribution as can be found in tensorflow.

### The best model

 **Stacked RNN + MLP:** We feed sentences from the stories into a two layer stacked LSTM RNN with hidden layer of size 50. This produces fixed length embeddings for each of the sentences. We concatenate these embeddings into a single vector that is sent to two layer MLP (multilayer perceptron) layer the predicts the order of the stentences for each story. We used  200 units for the hidden layer.  Our final model includes a modified version of LSTM accounting for layer normalisation (see comments in code above or 'experimentation' section below).

When compared to the provided model, our LSTM encoder replaces summation over the words in the sentences and our MLP replaces the single layer classifier in the original model.


### Other models 

* **Sentence Ordering using RNN**  We implemented the state-of-the-art model (Logeswaran et al 2016) [4], however it did not performed bettern then our Stacked RNN + ML model. The code is provided below.

* **Bi-directional RNN: ** Implemented but not included in the final model; it did not improve performance.

* **Attention, conditional encoding, seq-2-seq: ** We modified the **seq2seq** module. Did not improve the performance of a stacked LSTM. We note the attention model works worse than simple seq2seq; this is due to a conceptual fault in the attention mechanism.  

### Training procedure

* **Optimiser choice:** Adam and Nesterov momentum optimisation gave best performance, with the latter requiring shorter timesteps and more time to converge. 

* **Gradient clipping:** There were no improvements in performance. We attempted clipping gradients to specified min and max thresholds, as well as clipping to a maximum L2-norm chosen in the range [0.5, 1, ?, 10].

* **Regularization (L2): ** Adding L2 regularisation has not improved performance. 

* We shuffled the sentences within the stories and also the order of the stories during training.

* **Dropout:** For both MLP,  LSTM. The final model makes use of a global dropout of 0.9.


### Model selection

* **Early stopping: ** After monitoring the train/dev accuracy, we decide to stop the training procedure after 10-15 epochs (using the Adam optimiser). In this way overfitting is prevented. 10-15 epochs are sufficiently 'good' as a 'sweet spot' and also not too computationally expensive. 

### Hyperparameter optimisation

* ** Tuning (batch size, LSTM layer size, learning rate, fc_size, dropout, etc): ** We attempted to tune the learning rate using grid search, starting from log-space, and then linear-space. The batch size is tuned in a range of power of 2s. 

* ** Learning rate: ** Using a decaying learning rate didn?t improve performance. We tried using polynomial decay, starting with a value which yielded promising results and using various decay steps and polynomial powers. These experiments did not improve performance. 

### Experimentation

* ** Layer normalisation: ** 
In final model, we make use of the function LayerNormalizedLSTMCell. This function has been adapted from [1]. Layer normalisation is a feature recently published by Lei Ba et al. (2016) [2]. 

### Error Analysis
During model training, we monitored our train/test errors using Tensorboard.
After predictions, we observed frequencies of the number of mistakes made in the ordering of each of the 5 sentences for all stories. On the dev data, these amounted to 35, 219, 511, 579, 301, 226 story instances where we made 0,1,?5 mistakes, respectively. 
Additionally, we found that some errors were due to assigning the same order index to two sentences in a story, such as [3,1,0,1,4]. 
Other errors were understandable in cases where the misclassification was between two sentences which could easily be interchanged without affecting the meaning behind the story. An example of this is story index 300 from the dev set.

### References 
[1] http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
[2] Layer normalisation (Lei Ba et al 2016) https://arxiv.org/abs/1607.06450
[3] Sentence Ordering using RNN (Logeswaran et al 2016)
[4] Recurrent Model for Visual attention (Mnih et al 2014)