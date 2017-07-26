## My Features

The implemented features can be divided into several types. 

* **Token** features. These involve counting the number of occurrences of the following attributes: POS, stem, word and the 'left-over' part of the word after stemming (carried out by the child_word_stem_diff function). Other promising features involve counting character bigrams for a trigger word (bigrams function) or character trigrams for a trigger stem (trigrams_stem) function. Counting the number of words or characters in a token e.g. word_length or child_length produced more modest increases in F1. So did counting occurrences of tokens greater or shorter than a given length e.g stem_word_number, parent_word_number, parent_long_words.  

* **Content** features. These involve inspecting if there is a capitalised letter, double/triple letter, digits or hyphens in each token and counting these occurrences. These did not generally result in a significant increase of the F1 score, except in the case of counting digits occurrences for the syntactic child of a trigger word (included in the add_dependency_child_feats_modified function) 

* **Dependency** features. These involve for example, analysing attributes of syntactic children and parents of trigger words. Adding features for mapping a trigger word and its child, similarly the child and the grandchild, etc...showed particular promise. Generally, implementing our features on a syntactic child of a trigger word than on the trigger word itself gives better results (e.g. add_dependency_grandchild_feats).

* **Protein** features. Some of the most significant gains in F1 were accomplished by counting the number of words which are proteins in event candidates (see the protein_count function). These features showed plenty of synergy when implemented within child, grandchild, etc. counting the syntactic children/grandchildren that are proteins. (see child_protein_count, grandchild_protein_count, greatgrandchild_protein_count, also parent_protein_count).     

* **Sentence** features. Besides counting character bigram types we also count word bigram/trigram types with the word_bigrams and word_trigrams functions. These counts were particularly effective. 

The 'C' value of the logistic regression function was set to 1.8 after offline parameter optimisation. Also for this function, the class weight was set to 'balanced': this is important because it adjusts the regression weights inversely proportional to class frequencies, minimising the effect of the 'Nones'.


## Results

An F1 score of 67.93% was obtained (precision 65.76%, recall 70.24%). Development and test datasets were reguarly changed over to check the model worked well given different input (it usually dropped 1.5-2% F1).

** Error analysis **: Inspection of the confusion matrix reveals that the most mislabelling corresponds to misclassifying positive regulation events as 'None' and misclassifying 'None' as positive regulation (both have values of 200). Bare in mind that positive regulation events are the most frequent; less frequent events eg. transcription and regulation have lower recall, precision and F1 scores. ** Improvements **: A greater use of token span and dependency information could have been used to attain a 'deeper' model; for example using 'shortest path' features to such as N-grams of dependencies, or token features of tokens in a given path. More sophisticated developments could be implemented in Tensorflow using neural networks for example. 