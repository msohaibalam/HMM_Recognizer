import warnings
from asl_data import SinglesData
import numpy as np


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    df = test_set.df
    d_Xlengths = test_set._hmm_data
    d_seqs = {}
    d_seqlengths = {}
    for word in test_set.wordlist:
      # initialize an empty lists/dictionaries for each word
      d_logLs = {}
      d_seqs[word] = []
      d_seqlengths[word] = []
      # fetch all sequences for this word
      dd = df.loc[df['word'] == word]
      for ind in dd.index:
        d_seqs[word].append(d_Xlengths[ind][0])
        d_seqlengths[word].append(d_Xlengths[ind][1])
      # print("d_seqs[word]", d_seqs[word])
      # print("d_seqlengths[word]", d_seqlengths[word])
      d_seqs[word] = np.concatenate(d_seqs[word])
      d_seqlengths[word] = [i[0] for i in d_seqlengths[word]]
      # For every word in the test_set, d_seqs[word] and d_seqlengths[word] are X and Length respectively
      for someword, somemodel in models.items():
        try:
          d_logLs[someword] = somemodel.score(d_seqs[word], d_seqlengths[word])
        except (ValueError, AttributeError) as e:
          if 'rows of transmat_ must sum to 1.0' in e.args[0]:
            # according to the discussions in the slack channel, this seems to be a bug in the library
            # and should just be exceptioned out
            continue
          elif "'NoneType' object has no attribute 'score'" in e.args[0]:
            # this can occur if no model was fit and None was instead returned
            continue
      # append to probabilities list
      probabilities.append(d_logLs)
      # formulate best guess for the word
      try:
        best_guess = max(d_logLs, key=d_logLs.get)
      except ValueError as e:
        best_guess = ''
      guesses.append(best_guess)
    return probabilities, guesses
    # raise NotImplementedError
