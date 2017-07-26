%%capture
%load_ext autoreload
%autoreload 2
%matplotlib inline
#! SETUP 1 - DO NOT CHANGE, MOVE NOR COPY (RIEDEL)
import sys, os
_snlp_book_dir = "../../../../"
sys.path.append(_snlp_book_dir)
import math
from collections import defaultdict
import statnlpbook.bio as bio

#! SETUP 2 - DO NOT CHANGE, MOVE NOR COPY (RIEDEL)
train_path = _snlp_book_dir + "data/bionlp/train"
event_corpus = bio.load_assignment2_training_data(train_path)
event_train = event_corpus[:len(event_corpus)//4 * 3]
event_dev = event_corpus[len(event_corpus)//4 * 3:]
# event_train = event_corpus[len(event_corpus)//4 * 3:]
# event_dev = event_corpus[:len(event_corpus)//4 * 3]
assert(len(event_train)==53988)


# make example feature function
def add_dependency_child_feats(result, event):
    """
    Append to the `result` dictionary features based on the syntactic dependencies of the event trigger word of
    `event`. The feature keys should have the form "Child: [label]->[word]" where "[label]" is the syntactic label
    of the syntatic child (e.g. "det" in the case above), and "[word]" is the word of the syntactic child (e.g. "The"
    in the case above).
    Args:
        result: a defaultdict that returns `0.0` by default.
        event: the event for which we want to populate the `result` dictionary with dependency features.
    Returns:
        Nothing, but populates the `result` dictionary.
    """
    index = event.trigger_index       # features based on dependencies of event trigger word of `event'
    for child,label in event.sent.children[index]:
        result["Child: " + label + "->" + event.sent.tokens[child]['word']] += 1.0


# add more feature functions
import re

def add_dependency_child_feats_modified(result, event):
    '''
    Consists of multilple features for the syntactic children of a given trigger word.
    '''
    index = event.trigger_index
    for child, label in event.sent.children[index]:
        # Counts number of different "Child: [label]->[word]" combinations
        result["Child: " + label + "->" + event.sent.tokens[child]['word']] += 1.0
        # Features checking for number of words of the syntactic children  of the trigger word
        if (event.sent.tokens[child]['end'] - event.sent.tokens[child]['begin'] > 3):
            # counts variability wrt the child's stem attribute
            result["Child w/ more than 3 tokens (stem variability): " + event.sent.tokens[child]['stem']] += 1.0
            # counts variability wrt the child's pos attribute
            result["Child w/ more than 3 tokens (pos variability): " + event.sent.tokens[child]['pos']] += 1.0
            # if the syntactic child contains a number - counts word  attribute variability
            if re.findall("\w*\d\w*", event.sent.tokens[child]['word']):
                result['Child w/ digits (word variability): ' + event.sent.tokens[child]['word']] += 1.0
                # checks if the syntactic child has more than 7 characters
                if (len(event.sent.tokens[child]['word']) > 7):
                    result['Child w/ more than 7 characters (word var): ' + event.sent.tokens[child]['word']] += 1.0
            else:
                if (len(event.sent.tokens[child]['word']) > 7):
                    result['Child w/ more than 7 characters (word var): ' + event.sent.tokens[child]['word']] += 1.0
        # alternatively
        else:
            if re.findall("\w*\d\w*", event.sent.tokens[child]['word']):
                result['Child w/ digits (word variability): ' + event.sent.tokens[child]['word']] += 1.0
                if (len(event.sent.tokens[child]['word']) > 7):
                    result['Child w/ more than 7 characters (word var): ' + event.sent.tokens[child]['word']] += 1.0


def add_dependency_grandchild_feats(result, event):
    """
    Append to the `result` dictionary features based on the syntactic grandchild of the event trigger word of
    `event`. The feature are now in the form "Grandchild: [label]->[word]" where "[label]" is the syntactic label
    of the syntatic grandchild and "[word]" is the word of the syntactic grandchild.
    """
    index = event.trigger_index
    for child, label in event.sent.children[index]:
        index = child  # more distant syntactic dependencies can be found recursively.
        for grandchild, label in event.sent.children[index]:
            result["Grandchild: " + label + "->" + event.sent.tokens[grandchild]['word']] += 1.0
            # Additional features checking for the number of words in syntactic grandchildren (word)
            if (event.sent.tokens[grandchild]['end'] - event.sent.tokens[grandchild]['begin'] > 3):
                # counts variability wrt the grandchild's stem attribute
                result["Grandchild w/ more than 3 tokens (stem variability): " + event.sent.tokens[grandchild][
                    'stem']] += 1.0
                # counts variability wrt the grandchild's pos attribute
                result["Grandchild w/ more than 3 tokens (pos variability): " + event.sent.tokens[grandchild][
                    'pos']] += 1.0


def add_dependency_greatgrandchild_feats(result, event):
    """
    Append to the `result` dictionary features based on the syntactic greatgrandchild of the event trigger word of
    `event`. The feature is in the form "Greatgrandchild: [label]->[word]" where "[label]" is the syntactic label
    of the syntatic greatgrandchild and "[word]" is the word of the syntactic greatgrandchild.
    """
    index = event.trigger_index
    for child, label in event.sent.children[index]:
        index = child
        for grandchild, label in event.sent.children[index]:
            index = grandchild
            for greatgrandchild, label in event.sent.children[index]:
                result["Greatgrandchild: " + label + "->" + event.sent.tokens[greatgrandchild]['word']] += 1.0


def add_dependency_parent_feats(result, event):
    """
    Append to the `result` dictionary features based on the syntactic parent of the event trigger word of
    `event`. The feature is in the form "Parent: [label]->[word]" where "[label]" is the syntactic label
    of the syntatic parent and "[word]" is the word of the syntactic parent.
    """
    index = event.trigger_index
    # word attribute variability
    for parent, label in event.sent.parents[index]:
        result["Parent " + label + "->" + event.sent.tokens[parent]['word']] += 1.0


def bigrams(result, event):
    """
    Feature checking counting for occurrence of different character bigrams in the trigger word
    """
    index = event.trigger_index
    n = 2  # bigram
    for i in range(len(event.sent.tokens[index]['word']) - n + 1):
        result["Character bigram count: " + event.sent.tokens[index]['word'][i:i + n]] += 1.0


def trigrams_stem(result, event):
    """
    Feature checking counts for occurrence of different character trigrams in the trigger stem
    """
    index = event.trigger_index
    n = 3
    index = event.trigger_index
    for i in range(len(event.sent.tokens[index]['stem']) - n + 1):
        result["Character trigram count (stem): " + event.sent.tokens[index]['stem'][i:i + n]] += 1.0


def word_bigrams(result, event):
    """
    Feature checking counting for occurrence of different word bigrams at both sides of trigger word
    """
    index = event.trigger_index
    if (index > 0):
        result["Word bigram count (word var): " + event.sent.tokens[index]["word"] + event.sent.tokens[index - 1][
            "word"]] += 1.0
        result["Word bigram count (stem var): " + event.sent.tokens[index]["stem"] + event.sent.tokens[index - 1][
            "stem"]] += 1.0
    if (index < len(event.sent.tokens) - 4):  # crude termination condition
        result["Word bigram count - reverse order (word var): " + event.sent.tokens[index]["word"] \
               + event.sent.tokens[index + 1]["word"]] += 1.0
        result["Word bigram count - reverse order (word var): " + event.sent.tokens[index]["stem"] \
               + event.sent.tokens[index + 1]["stem"]] += 1.0


def word_trigrams(result, event):
    """
    Feature checking counting for occurrence of different word trigrams for the trigger (3 possible arrangements)
    """
    index = event.trigger_index
    if (index > 1):
        result["Word trigram count (word var) - arrangement 1: " + event.sent.tokens[index]["word"] \
               + event.sent.tokens[index - 1]["word"] + event.sent.tokens[index - 2]["word"]] += 1.0
        result["Word trigram count (stem var) - arrangement 1: " + event.sent.tokens[index]["stem"] \
               + event.sent.tokens[index - 1]["stem"] + event.sent.tokens[index - 2]["stem"]] += 1.0
    if (index < len(event.sent.tokens) - 4):  # crude termination condition
        result["Word trigram count - arrangement 2: " + event.sent.tokens[index]["word"] \
               + event.sent.tokens[index + 1]["word"] + event.sent.tokens[index + 2]["word"]] += 1.0
    if ((index < len(event.sent.tokens) - 4) & (index > 1)):  # crude termination condition
        result["Word trigram count - arrangement 3: " + event.sent.tokens[index]["word"] \
               + event.sent.tokens[index - 1]["word"] + event.sent.tokens[index + 1]["word"]] += 1.0


def word_length(result, event):
    """
    Feature checking for variability in the character length of a given trigger word and its stem.
    """
    index = event.trigger_index
    result["Trigger word character length: " + str(len(event.sent.tokens[index]["word"]))] += 1.0
    result["Trigger stem character length: " + str(len(event.sent.tokens[index]["stem"]))] += 1.0


def child_length(result, event):
    """
    Feature checking for variability in the character length of the syntactic child of a trigger word.
    """
    index = event.trigger_index
    for child, label in event.sent.children[index]:
        result["Child character length: " + str(len(event.sent.tokens[child]["word"]))] += 1.0


def __sub__(word, stem):
    """
    Function used in the next feature to calculate the difference between a word and its stem
    """
    return word.replace(stem, '', 1)


def child_word_stem_diff(result, event):
    """
    Feature checking for variability in the difference between the word and the stem attributes for a syntactic child.
    """
    index = event.trigger_index
    for child, label in event.sent.children[index]:
        result["Word/stem difference (child): " \
               + __sub__(event.sent.tokens[child]['word'], event.sent.tokens[child]['stem'])] += 1.0


def protein_count(result, event):
    """
    Feature checking for variability in the protein counts for each event
    """
    index = event.trigger_index
    cont = 0
    for i in event.sent.is_protein.values():
        if i:
            cont += 1
    result['Protein count: ' + str(cont)] += 1.0


def child_protein_count(result, event):
    """
    Protein count for children
    """
    index = event.trigger_index
    cont = 0
    for child, label in event.sent.children[index]:
        if event.sent.is_protein[child]:
            cont += 1
    result["Child protein count: " + str(cont)] += 1.0


def parent_protein_count(result, event):
    """
    Protein count for parent
    """
    index = event.trigger_index
    cont = 0
    for parent, label in event.sent.parents[index]:
        if event.sent.is_protein[parent]:
            cont += 1
    result["Parent protein count: " + str(cont)] += 1.0


def grandchild_protein_count(result, event):
    """
    Protein count for grandchild
    """
    index = event.trigger_index
    cont = 0
    for child, label in event.sent.children[index]:
        index = child
        for grandchild, label in event.sent.children[index]:
            if event.sent.is_protein[grandchild]:
                cont += 1
    result["Grandchild protein count: " + str(cont)] += 1.0


def greatgrandchild_protein_count(result, event):
    """
    Protein count for greatgrandchild
    """
    index = event.trigger_index
    cont = 0
    for child, label in event.sent.children[index]:
        index = child
        for grandchild, label in event.sent.children[index]:
            index = grandchild
            for greatgrandchild, label in event.sent.children[index]:
                if event.sent.is_protein[greatgrandchild]:
                    cont += 1
    result["Greatgrandchild protein count: " + str(cont)] += 1.0


def greatgreatgrandchild_protein_count(result, event):
    """
    Protein count for greatgreatgrandchild
    """
    index = event.trigger_index
    cont = 0
    for child, label in event.sent.children[index]:
        index = child
        for grandchild, label in event.sent.children[index]:
            index = grandchild
            for greatgrandchild, label in event.sent.children[index]:
                index = greatgrandchild
                for greatgreatgrandchild, label in event.sent.children[index]:
                    if event.sent.is_protein[greatgreatgrandchild]:
                        cont += 1
    result["Greatgreatgrandchild protein count: " + str(cont)] += 1.0


def stem_word_number(result, event):
    index = event.trigger_index
    # Feature checking for the number of tokens in trigger word
    if (event.sent.tokens[index]['end'] - event.sent.tokens[index]['begin'] > 3):
        result["Stem w\ trigger word of more than 3 words: " + event.sent.tokens[index]['stem']] += 1.0


def parent_word_number(result, event):
    index = event.trigger_index
    # Feature counting variability in number of words with only one token in the syntactic parent of trigger
    # stem variability
    for parent, label in event.sent.parents[index]:
        if (event.sent.tokens[parent]['end'] - event.sent.tokens[parent]['begin'] < 2):
            result["One token word (parent): " + event.sent.tokens[parent]['stem']] += 1.0


def parents_long_words(result, event):
    index = event.trigger_index
    # Feature counting variability in long words (more than 13 characters) in the syntactic parent
    for parent, label in event.sent.parents[index]:
        if (len(event.sent.tokens[parent]['word']) > 13):
            result['Parent - words w\ more than 13 chars: ' + event.sent.tokens[parent]['word']] += 1.0


### MODEL EVALUATION
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

# converts labels into integers, and vice versa, needed by scikit-learn.
label_encoder = LabelEncoder()

# encodes feature dictionaries as numpy vectors, needed by scikit-learn.
vectorizer = DictVectorizer()

def event_feat(event):
    """
    This feature function returns a dictionary representation of the event candidate. You can improve the model
    by improving this feature function.
    Args:
        event: the `EventCandidate` object to produce a feature dictionary for.
    Returns:
        a dictionary with feature keys/indices mapped to feature counts.
    """
    result = defaultdict(float)
    result['trigger_word=' + event.sent.tokens[event.trigger_index]['word']] += 1.0 # word occurrence
    result['trigger_POS=' + event.sent.tokens[event.trigger_index]['pos']] += 1.0 # pos occurrence
    result['trigger_stem=' + event.sent.tokens[event.trigger_index]['stem']] += 1.0 # stem occurence
    add_dependency_child_feats_modified(result, event)
    stem_word_number(result,event)
    add_dependency_parent_feats(result, event)
    parents_long_words(result,event)
    parent_word_number(result,event)
    trigrams_stem(result,event)
    word_bigrams(result,event)
    word_trigrams(result,event)
    word_length(result,event)
    child_length(result,event)
    bigrams(result,event)
    protein_count(result,event)
    child_protein_count(result,event)
    grandchild_protein_count(result,event)
    greatgrandchild_protein_count(result,event)
    greatgreatgrandchild_protein_count(result,event)
    parent_protein_count(result,event)
    add_dependency_grandchild_feats(result,event)
    add_dependency_greatgrandchild_feats(result,event)
    child_word_stem_diff(result,event)
    return result

# We convert the event candidates and their labels into vectors and integers, respectively.
train_event_x = vectorizer.fit_transform([event_feat(x) for x,_ in event_train])
train_event_y = label_encoder.fit_transform([y for _,y in event_train])

# Create and train the model. Feel free to experiment with other parameters and learners.
lr = LogisticRegression(C=1.8, class_weight='balanced')
lr.fit(train_event_x, train_event_y)

def predict_event_labels(event_candidates):
    """
    This function receives a list of `bio.EventCandidate` objects and predicts their labels.
    It is currently implemented using scikit-learn, but you are free to replace it with any other
    implementation as long as you fulfil its contract.
    Args:
        event_candidates: A list of `EventCandidate` objects to label.
    Returns:
        a list of event labels, where the i-th label belongs to the i-th event candidate in the input.
    """
    event_x = vectorizer.transform([event_feat(e) for e in event_candidates])
    event_y = label_encoder.inverse_transform(lr.predict(event_x))
    return event_y

#! ASSESSMENT - DO NOT CHANGE, MOVE NOR COPY (RIEDEL)
_snlp_event_test = event_dev # This line will be changed by us after submission to point to a test set.
_snlp_event_test_guess = predict_event_labels([x for x,_ in _snlp_event_test[:]])
_snlp_cm_test = bio.create_confusion_matrix(_snlp_event_test,_snlp_event_test_guess)
bio.evaluate(_snlp_cm_test)[2] # This is the F1 score