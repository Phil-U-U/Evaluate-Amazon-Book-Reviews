'''
        Evaluate Amazon Book Reviews - Extract features by calling scanText
        Author: Phil H. Cui
        Time:   10/12/2016
'''

# ************************* Acquire Raw Data *************************

import scanCnt import scanText
import gzip
import pandas as pd

class getRawData( object ):

    def __init__( self,  file_name ):
        self.corpus = []
        self.N_sentences = []
        self.Avg_wordsPerSent = []
        self.IsBrackets = []
        self.N_words = []
        self.N_unique_words = []
        self.N_upperCases = []
        self.N_puncs = []
        self.N_puncs_unique = []

        self.helpful_ratio = []
        self.helpful_pair = []
        self.cntNoVotes = 0
        self.cntVotes_less_than_nVotes = 0
        self.summary = []

        self.parser = iter( self.parse() )
        self.st = scanText()

        self.file_name = file_name

        self.df = None

    # use generator to parse data
    def parse( self ):
        g = gzip.open(self.file_name, 'r')

        for l in g:
            yield eval(l)

    def read( self, sample_max = 100000, votes = 8 ):

        for cnt in xrange( sample_max ):

            review = self.parser.next()

            helpfulness = review['helpful']

            if helpfulness[1] >= votes:
                text = review['reviewText']
                n_sentences, avg_wordsPerSent, n_words, n_unique_words, n_upperCases, n_puncs, n_puncs_unique, isBrackets = self.st.scan(text)

                self.corpus.append(text)
                self.N_sentences.append( n_sentences )
                self.Avg_wordsPerSent.append( avg_wordsPerSent )
                self.IsBrackets.append( isBrackets )
                self.N_words.append( n_words )
                self.N_unique_words.append( n_unique_words )
                self.N_upperCases.append( n_upperCases )
                self.N_puncs.append( n_puncs )
                self.N_puncs_unique.append( n_puncs_unique )

                rating = float(helpfulness[0]) / helpfulness[1]
                self.helpful_ratio.append( rating )
                # self.helpful_pair.append( (helpfulness[0], helpfulness[1]) )

                # if rating < 0.5:
                #     print helpfulness[0], helpfulness[1]
                #     print text


            else:
                self.cntVotes_less_than_nVotes += 1

            if cnt == sample_max-1:

                self.df = pd.DataFrame({"corpus": self.corpus,
                                        "Helpfullness": self.helpful_ratio,
                                        # "helpful_pair": self.helpful_pair,
                                        "N_sentences": self.N_sentences,
                                        "Avg_wordsPerSent": self.Avg_wordsPerSent,
                                        "IsBrackets": self.IsBrackets,
                                        "N_words": self.N_words,
                                        "N_unique_words": self.N_unique_words,
                                        "N_upperCases": self.N_upperCases,
                                        "N_puncs": self.N_puncs,
                                        "N_puncs_unique": self.N_puncs_unique
                                        })
                break

        print "Votes ratio( >" + str(votes) + " vs all): {}".format(1-self.cntVotes_less_than_nVotes/float(sample_max))
        return self.df
