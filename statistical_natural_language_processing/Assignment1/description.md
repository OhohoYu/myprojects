## The Model

The implemented language model utilises the interpolated Kneser-Ney smoothing<sup>1,2</sup> method. Kneser-Ney is a discounting method; discounting counts for more frequent N-grams is necessary to save some probability mass for the unseen N-grams. In this case, a simple implementation which interpolates a unigram and a bigram model has been developed. Kneser-Ney differs from simpler discounting methods (e.g. absolute discounting) in its treatment of unigram frequencies. Relying only on unigram frequencies to predict the frequencies of N-grams gives skewed results <sup>3</sup>. Kneser-Ney corrects this by introducing the term $P_{CONTINUATION}$, where:

\begin{equation}
P_{CONTINUATION}(w_i) \propto \vert\{w_{i-1}: c(w_{i-1}w_{i}) > 0 \}\vert. 
\end{equation}

The right hand side represents the number of word types $w_{i-1}$ seen to precede $w_{i}$. Whereas the non-modified unigram probability $p(w_i)$ seeks to answer ''how likely is $w_i$?'', $P_{CONTINUATION}(w_i)$ answers ''how likely is $w_i$ to appear as a novel continuation?''. $P_{CONTINUATION}(w_i)$ can then be normalised by the total number of word bigram types: 

\begin{equation}
P_{CONTINUATION}(w_i) = \frac{\vert \{ w_{i-1}: c(w_{i-1}w_{i}) > 0 \} \vert }{\vert \{ (w_{j-1}, w_j): c(w_{j-1}w_{j}) > 0) \} \vert} = \frac{\vert \{ w_{i-1}: c(w_{i-1}w_{i}) > 0 \} \vert }{\sum_{w_i'} \vert \{ w_{i-1}': c(w_{i-1}'w_i') > 0 \} \vert }.
\end{equation}

The definitive formulation of Interpolated Kneser-Ney is as follows: 
\begin{equation}
P_{KN}(w_i \vert w_{i-1}) = \frac{max(c(w_{i-1}w_{i})-d,0)}{c(w_{i-1})} + \lambda(w_{i-1})P_{CONTINUATION}(w_i),
\end{equation}

where $\lambda(w_{i-1})$ is a normalising constant; the probability mass we have discounted from the seen N-grams:

\begin{equation}
\lambda(w_{i-1}) = \frac{d}{c(w_{i-1})}\vert \{ w : c(w_{i-1}, w) > 0 \} \vert, 
\end{equation}

where $\frac{d}{c(w_{i-1})}$ is the normalised discount and the second term represents the number of word types that can follow $w_{i-1}$.
## The Code

To implement interpolated Kneser-Ney smoothing, the class `KneserNeyBigram` is introduced as a subclass of the `lm.LanguageModel` class. The following instance variables are initialised: 

| variable 	| corresponds to: 	              | 
|:---------------------:|:-----------------------------:| 
| `self._counts[(word,) + history]`		| $c(w_{i-1}w_{i})$	| 
| `self._norm[history]`		| $c(w_{i-1})$	| 
| `self._tot_bigrams['grams']`		| $\sum_{w_i'} \vert \{ w_{i-1}': c(w_{i-1}'w_i') > 0 \} \vert$ | 
| `self._follow_up[history]`	| $\vert \{ w : c(w_{i-1}, w) > 0 \} \vert$	| 
| `self._preceding[word]`		| $\vert\{w_{i-1}: c(w_{i-1}w_{i}) > 0 \}\vert$ | 
| `self._p_continuation[word]`		| $P_{CONTINUATION}(w_i)$ |
| `self._lambda[history]`		| $\lambda(w_{i-1})$ |
| `self._discount`		| $d$ | 

Note how within each `for` loop, for a bigram $p(w_i \vert w_{i-1})$, variables associated with $w_i$ are indexed by `word` and variables associated with $w_{i-1}$ are indexed by `history`. The first `for` loop loops over every pair of consecutive words in the dataset, updating $c(w_{i-1}w_{i})$ and $c(w_{i-1})$. For each pair, it also checks whether the word preceding $w_i$ and the word following $w_{i-1}$ is new, keeping track of the number of times a word $w_i$ appears as a novel continuation, the number of word types that can follow $w_{i-1}$ and the total number of word bigram types. Once the total `self._tot_bigrams['grams']` is known, we can loop over the dataset again and calculate $P_{CONTINUATION}(w_i)$ for each word $w_i$. At this point $\lambda(w_{i-1})$ is also calculated. 

The Kneser-Ney probabilities $P_{KN}(w_i \vert w_{i-1})$ are calculated within the `probability` method. The first two lines of this method ensure that the language model only operates on the defined vocabulary. To handle unseen bigrams $p(x_2 \vert x_1)$, I am "backing-off" to the probability of the last unigram $p(x_2)$, which under Kneser-Ney smoothing is $P_{CONTINUATION}(x_2)$.

Before calling `KneserNeyBigram`, `lm.inject_OOVs` is applied to the training set. This class estimates the probability of an out-of-vocabulary word by replacing the first encounter of each word in the training set with the OOV token. `KneserNeyBigram` is called within the `create_lm` function, being "wrapped" by the `lm.OOVAwareLM` class. `lm.OOVAwareLM` allows the language model to operate on the target vocabulary by mapping `vocab` words unseen by the training set to the OOV token. 

`self._discount` ($ 0 < d < 1 $) is set to 0.79 after running parameter optimisation offline. Note there is no need to optimise $\lambda(w_{i-1})$, which has been carefully defined to make the sum of conditional probabilities $p_{KN}(w_i \vert w_{i-1})$ equal to one.

## References
1. **Improved backing-off for M-gram language modeling.** Kneser, R. and Ney, H. (1995). In ICASSP-95, Vol. 1, pp. 181?184.
2. **An empirical study of smoothing techniques for language modeling.** Chen, S. F. and Goodman, J. (1998). Tech. rep. TR-10-98, Computer Science Group, Harvard University. 
3. [**Brown University: Introduction to Computational Linguistics.**][1] 
4. **Chapter 4: N-Grams.** Jurafsky, D. and Martin, J.H. (2016). In: Speech and Language Processing (3rd ed. draft).
5. [**Stanford University NLP Lunch Tutorial: Smoothing**][2] MacCartney, B (2005).

[1]: http://cs.brown.edu/courses/cs146/files/langmod_2015.pdf/
[2]: http://nlp.stanford.edu/~wcmac/papers/20050421-smoothing-tutorial.pdf