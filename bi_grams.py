import nltk
import re
import collections

from nltk.util import ngrams
from nltk.corpus import stopwords


def return_bi_grams(text):

    def initial_clean(text):
        text = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", text)
        text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
        text = re.sub("[^a-zA-Z ]", " ", text)
        text = re.sub(r'\d+', ' ', text)
        text = text.lower()
        text = nltk.word_tokenize(text)
        return text

    def rem_stop(text):
        return [word for word in text if word not in stop_words]

    stop_words = stopwords.words('english')
    sen = rem_stop(initial_clean(text))
    esBigrams = ngrams(sen, 2)
    esBigramFreq = collections.Counter(esBigrams)
    return esBigramFreq.most_common(10)