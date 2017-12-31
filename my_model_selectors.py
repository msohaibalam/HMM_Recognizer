import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # initialize variables to keep track of best performing model
        best_score = 999999999   # initialize to some really large value
        best_n = 0
        best_model = None
        # initialize empty dictionary, used to store lists of BIC scores for each n below
        d_BIC = {}
        for n in range(self.min_n_components, self.max_n_components + 1):

            try:
                # fit model and obtain logL probabilities
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)

                ### Excellent resource for # parameters estimation in HMM: ftp://metron.sta.uniroma1.it/RePEc/articoli/2002-LX-3_4-11.pdf
                ### via Slack channel discussion: https://ai-nd.slack.com/files/ylu/F4S90AJFR/number_of_parameters_in_bic.txt
                # With n states, we can construct an n x n matrix, with n^2 parameters. ( + n^2 )
                # But at least one value in each row is fixed in terms of the other values because probabilities must sum to 1,
                # so this removes n free parameters. ( - n )
                # With d features, there are d*n Gaussian means and d*n Gaussian variances to estimate. ( + 2*d*n )
                # Finally, the initial distribution for each state P(X1 = i) for i=1,...,n are also free parameters
                # that can be learned if not specified; with n features, these are n-1 free parameters
                # since probabilities must sum to 1 ( + n-1 )

                # the error check below is just to make sure self.X is in the correct format
                if len(np.array(self.X).shape) > 2:
                    raise ValueError("self.X is not truly concatenated")
                # the number of features can be extracted from the shape of self.X
                d = int(np.array(self.X).shape[1])

                # So in all, we have n^2 - n + 2*d*n + n - 1 = n*(n-1) + (n-1) + 2*d*n = (n+1)*(n-1) + 2*d*n
                #                           = n^2 + 2*d*n - 1   free parameters
                p = (n**2) + (2*d*n) - 1
                # number of data points
                N = len(self.X)
                # BIC score
                d_BIC[n] = - (2 * logL) + (p * np.log(N))
                # check if current logL score better than best one so far
                if d_BIC[n] < best_score:
                    best_score = d_BIC[n]
                    best_n = n

            except ValueError as e:
                # according to the discussions in the Slack channel, this seems to be a bug in the library
                # and should just be exceptioned out
                if 'rows of transmat_ must sum to 1.0' in e.args[0]:
                    continue

        # return the best model, or None if none was found
        if best_n != 0:
            best_model = GaussianHMM(n_components=best_n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        return best_model
        # TODO implement model selection based on BIC scores
        # raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        # initialize variables to keep track of best performing model
        best_score = - 999999999  # initalize to some really small value
        best_n = 0
        best_model = None
        d_DIC = {}
        for n in range(self.min_n_components, self.max_n_components + 1):

            try:
                # first, calculate the logL for this_word
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL_this_word = model.score(self.X, self.lengths)
                # initialize logL for all other words to zero
                logL_other_words = []
                for other_word in list(set(self.words) - set([self.this_word])):
                    other_X, other_lengths = self.hwords[other_word]
                    try:
                        model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(other_X, other_lengths)
                        logL_other_word = model.score(other_X, other_lengths)
                        logL_other_words.append(logL_other_word)
                    except ValueError as e:
                        # according to the discussions in the Slack channel, this seems to be a bug in the library
                        # and should just be exceptioned out
                        if 'rows of transmat_ must sum to 1.0' in e.args[0]:
                            pass
                # calculate the avg logL of all other words
                avg_logL_other_words = np.mean(logL_other_words)
                # DIC score
                d_DIC[n] = logL_this_word - avg_logL_other_words
                # check if current DIC score better than best one so far
                if d_DIC[n] > best_score:
                    best_score = d_DIC[n]
                    best_n = n

            except ValueError as e:
                # according to the discussions in the Slack channel, this seems to be a bug in the library
                # and should just be exceptioned out
                if 'rows of transmat_ must sum to 1.0' in e.args[0]:
                    continue

        # return the best model, or None if none was found
        if best_n != 0:
            best_model = GaussianHMM(n_components=best_n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        return best_model
        # raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection using CV
        # initialize variables to keep track of best performing model
        best_score = - 999999999  # initalize to some really small value
        best_n = 0
        best_model = None
        # initialize suitable split method
        split_method = KFold(n_splits=min(max(2, len(self.sequences)), 5))
        # initialize empty dictionary, used to store lists of logL scores for each n below
        d_logL = {}
        for n in range(self.min_n_components, self.max_n_components + 1):
            # initalize empty list for each n
            d_logL[n] = []
            # cannot train/test split single-item sequences, ignore these
            if len(self.sequences) == 1:
                continue
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                # create test and train cross-validation sets
                X_train, Length_train = combine_sequences(list(cv_train_idx), self.sequences)
                X_test, Length_test = combine_sequences(list(cv_test_idx), self.sequences)
                try:
                    # fit the model and obtain the logL score
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, Length_train)
                    logL = model.score(X_test, Length_test)
                except ValueError as e:
                    if 'rows of transmat_ must sum to 1.0' in e.args[0]:
                        # according to the discussions in the slack channel, this seems to be a bug in the library
                        # and should just be exceptioned out
                        continue
                    elif ('n_samples=' in e.args[0]) and ('should be' in e.args[0]) and ('>= n_clusters' in e.args[0]):
                        # according to the discussions in the slack channel, this too seems to be a bug in the library
                        # and should just be exceptioned out
                        continue
                    else:
                        print("Unrecognized ValueError")
                        raise
                # append the logL score for this n, to be able to compute mean after the loop ends
                d_logL[n].append(logL)
            # check if current logL score better than best one so far
            if np.mean(d_logL[n]) > best_score:
                best_score = np.mean(d_logL[n])
                best_n = n
        # return the best model, or None if none was found
        if best_n != 0:
            best_model = GaussianHMM(n_components=best_n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        return best_model
