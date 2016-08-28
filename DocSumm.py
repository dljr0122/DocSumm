# !/usr/bin/python

# from __future__ import division
from __future__ import print_function
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from collections import defaultdict

import Algorithms as Alg
import datetime as dt
import string

import nltk  # , re, pprint
import argparse

punctuations = ['.', '?', '"', '\'', '\'\'', '!', ',', ';', ':', '/', '\\', '`',
                '``', '_', '(', ')', '[', ']', '{', '}', '<', '>']
debug_on = False
timer = False
duplicates = 0


class DocParse:
    max_sentence = 10000
    include_file = ''
    outfile = ''
    count = 0

    def __init__(self):
        self.stemming_on = False
        self.stop_word_on = False
        self.summary = False
        self.use_threshold = False
        self.max_words_in_summary = 100
        self.keep_all = True  # by default do not exclude dupe words
        self.normalize = False
        self.score = 'size'  # size | tfidf | stfidf
        self.update = False
        self.total_sentences = 0
        self.sentence_dictionary = defaultdict(list)  # map of modified sentences to actual sentences (tokenized)

        self.dictionary = {}
        # keys are final tokenized output
        # values are 2-tuple of original sentence and size

        self.mod_words = ()  # all unique words of document
        self.mod_sentences = ((),)
        self.unique_sent = ((),)
        self.alg = Alg.Algorithms()
        self.stemmer = EnglishStemmer()
        self.doc_size = 0

    def tokenize(self, in_file):
        """Reads in_file and tokenizes into words."""

        global debug_on
        global punctuations
        if debug_on: print('stem:', self.stemming_on)
        if debug_on: print('stop:', self.stop_word_on)
        if debug_on: print('keep:', self.keep_all)
        f = open(in_file)
        raw = f.read()
        sentences_list = []
        words_list = []
        dictionary_values = []
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw_sentences = sent_tokenizer.tokenize(raw)
        self.total_sentences = len(raw_sentences)

        # operate on each sentence string from in_file
        for s, sentence in enumerate(raw_sentences):
            if debug_on: print('sentence #', str(s + 1))

            # remove newlines, after tokenizer.
            sentence = sentence.replace('\n', ' ')
            # change sentence into tokens (words)
            tokens = nltk.word_tokenize(sentence)
            # create table that maps all punctuation marks to None
            table = str.maketrans({key: None for key in string.punctuation if key != '\''})
            # keep only words and numbers
            words = [word.lower() for word in tokens if (word.translate(table) and word != '\'')]
            if debug_on:
                print("nltk tokens", end=":")
                print(tokens)
                print("parsed words", end=": ")
                print(words)
                print(len(words))
            sentence_size = len(words)
            if debug_on: print('sent len:', str(sentence_size))
            # remove stop words
            if self.stop_word_on:
                filtered_words = [word for word in words if word not in stopwords.words('english')]
                words = filtered_words
            # stem words
            if self.stemming_on:
                filtered_words = [self.stemmer.stem(word) for word in words]
                words = filtered_words
            if debug_on: print('after filters:', str(words))
            # compress sentences to unique words only if not doing greedy3 or tf-idf
            if self.keep_all:
                unique_words = words
                # removes repeated sentences
                if words not in sentences_list:
                    sentences_list.append(words)
                    dictionary_values.append((sentence, sentence_size, s))
            else:
                # make list of unique words from current sentence
                unique_words = list(set(words))
                # if unique word set not in sentence list than add this set
                # all repeated sentences will be removed at this stage
                if unique_words not in sentences_list:
                    sentences_list.append(unique_words)
                    # update local dictionary that maps index to tuple (original sentence, and length)
                    dictionary_values.append((sentence, sentence_size, s))
            if debug_on: print(sentences_list)

            # add unique words to doc word list
            for w, word in enumerate(words):
                if word not in words_list:
                    words_list.append(word)

            # add the modified sentence into dictionary
            self.sentence_dictionary[tuple(unique_words)].append(sentence)

        # this loop changes all the sentences of sentence_list into tuples
        for s, sentence in enumerate(sentences_list):
            sentences_list[s] = tuple(sentence)
            self.dictionary[sentences_list[s]] = dictionary_values[s]

        # store word list as tuple
        # store sentence list as tuple
        self.mod_words = tuple(words_list)
        self.mod_sentences = tuple(sentences_list)
        self.doc_size = len(self.mod_sentences)

    def find_dominating_set(self, option='greedy'):
        if option == 'greedy':
            if self.score == 'size':
                if self.normalize:
                    self.do_g_unique()
                else:
                    self.do_g_size()
            elif self.score == 'tfidf':
                self.do_g_tfidf()
            elif self.score == 'stfidf':
                self.do_g_stfidf()
        elif option == 'dynamic':
            self.do_dynamic()
        elif option == 'optimal':
            self.do_bottomup()
        elif option == 'mcdonald':
            self.do_mcdonald()

    def do_mcdonald(self):
        global debug_on
        answer = self.alg.mcdonald(self.mod_sentences, self.mod_words, self.dictionary,
                                   use_threshold=self.use_threshold, word_count=self.max_words_in_summary)
        print(answer)

    def do_g_size(self):
        global debug_on
        answer = self.alg.greedy(self.mod_sentences, self.mod_words, self.dictionary,
                                 update=self.update,
                                 word_threshold=self.use_threshold,
                                 word_count=self.max_words_in_summary)
        if debug_on: print('greedy answer', answer)
        if self.summary:
            print(self.make_summary(answer))
        else:
            print('len(ans):', len(answer))
            print('len(doc):', self.total_sentences)
        if debug_on:
            print('*****')
            print(self.sentence_dictionary)
            print('*****')

    def do_g_unique(self):
        global debug_on
        answer = self.alg.greedy2(self.mod_sentences, self.mod_words, self.sentence_dictionary, self.dictionary,
                                  update=self.update,
                                  word_threshold=self.use_threshold,
                                  word_count=self.max_words_in_summary)
        if debug_on: print('greedy answer', answer)
        if self.summary:
            print(self.make_summary(answer))
        else:
            print('len(ans):', len(answer))
            print('len(doc):', self.total_sentences)
        if debug_on:
            print('*****')
            print(self.sentence_dictionary)
            print('*****')

    def do_g_tfidf(self):
        global debug_on
        answer = self.alg.tfidf(self.mod_sentences, self.mod_words, self.dictionary,
                                ratio=self.normalize,
                                update=self.update,
                                word_count=self.max_words_in_summary,
                                use_threshold=self.use_threshold)
        if debug_on: print('tfidf answer', answer)
        if self.summary:
            print(self.make_summary(answer))
        else:
            print('len(ans):', len(answer))
            print('len(doc):', self.total_sentences)
            if debug_on:
                print('*****')
                print(self.sentence_dictionary)
                print('*****')

    def do_g_stfidf(self):
        global debug_on
        answer = self.alg.stfidf(self.mod_sentences, self.mod_words, self.dictionary,
                                 update=self.update,
                                 ratio=self.normalize,
                                 word_count=self.max_words_in_summary,
                                 use_threshold=self.use_threshold)
        if debug_on: print('tfidf answer', answer)
        if self.summary:
            print(self.make_summary(answer))
        else:
            print('len(ans):', len(answer))
            print('len(doc):', self.total_sentences)
            if debug_on:
                print('*****')
                print(self.sentence_dictionary)
                print('*****')

    def do_g_rtfidf(self):
        global debug_on
        answer = self.alg.tfidf(self.mod_sentences, self.mod_words, self.dictionary, ratio=True,
                                use_threshold=self.use_threshold)
        if debug_on: print('tfidf answer', answer)
        if self.summary:
            print(self.make_summary(answer))
        else:
            print('len(ans):', len(answer))
            print('len(doc):', self.total_sentences)
            if debug_on:
                print('*****')
                print(self.sentence_dictionary)
                print('*****')

    def do_bottomup(self):
        global debug_on
        self.alg.bottom_up(self.mod_sentences)
        if self.summary:
            print(self.make_summary(self.alg.dynamic_ans))
        else:
            print('len(ans):', len(self.alg.dynamic_ans))
            print('len(doc):', self.total_sentences)

    def do_dynamic(self):
        global debug_on
        if debug_on: print(self.mod_sentences)
        if debug_on: print(self.mod_words)
        if self.doc_size > 20:
            print('too many sentences:', self.doc_size)
            return
        # else:
        #     print('there are', self.doc_size, 'sentences')
        self.alg.dynamic(self.mod_sentences, self.mod_words)
        # self.sd.dynamic_lookup(set_of_sents, set_of_words)
        if debug_on: print('')
        self.alg.dynamic_calc_answer(self.mod_sentences, self.mod_words)
        if debug_on: print(self.alg.dynamic_ans)
        if debug_on:
            for i, items in enumerate(self.alg.dynamic_ans):
                print(i, ":", items)
        if self.summary:
            print(self.make_summary(self.alg.dynamic_ans))
        else:
            print('len(ans):', len(self.alg.dynamic_ans))
            print('len(doc):', self.total_sentences)
        # print 'dynamic answer', answer
        pass

    def make_summary(self, sentences):
        global debug_on
        ret_val = []
        word_count = 0
        for sentence in sentences:
            if self.dictionary[sentence][1] <= (self.max_words_in_summary - word_count) or \
                            self.max_words_in_summary == 0:
                if debug_on: print(str(self.dictionary[sentence][1]) + ": " + self.dictionary[sentence][0])
                ret_val.append(self.dictionary[sentence][0])
                word_count += self.dictionary[sentence][1]
            else:
                if debug_on: print(str(self.dictionary[sentence][1]) + ": " + self.dictionary[sentence][0])
                ret_val.append(self.shorten(self.dictionary[sentence][0], self.max_words_in_summary - word_count))
                break
                pass
        if self.outfile:
            with open(self.outfile, 'w') as f:
                f.write(" ".join(ret_val))
            pass
        return " ".join(ret_val)
        pass

    def shorten(self, sentence, length):
        global punctuations
        global debug_on
        tokens = nltk.word_tokenize(sentence)
        # remove all non-alphanumeric characters
        words_used = 0
        words = []
        for word in tokens:
            if words_used == length:
                break
            if debug_on: print(word, end=' ')
            words.append(word)
            if word not in punctuations:
                words_used += 1
                if debug_on: print('keep', words_used)
            else:
                if debug_on: print('remove')
        # words = [word for word in tokens if word not in punctuations]
        # return " ".join(words[:length-len(words)])
        return " ".join(words)


def main():
    global debug_on
    global timer
    before = dt.datetime.now()
    parser = argparse.ArgumentParser(description='Graph from Text')
    parser.set_defaults(run='greedy', score='size')
    parser.add_argument("infile", help='name of input file')
    parser.add_argument("-r", "--run", metavar='[greedy|optimal|mcdonald]',
                        choices=['greedy', 'optimal', 'mcdonald'],
                        help='finds dominating set from sentences of document')
    parser.add_argument("-c", "--score",
                        choices=['size', 'tfidf', 'stfidf'],
                        default='size')
    # unique word option.
    parser.add_argument("-d", "--distinct", action='store_true', help='reduce sentences to distinct words')
    # normalize sentence
    parser.add_argument("-n", "--normalize", action='store_true', help='normalize score by sentence size')
    # sets parser to do stemming
    parser.add_argument("-s", "--stem", action='store_true', help='turns on stemming function')
    # option to give a word threshold for the summary.
    parser.add_argument("-t", "--threshold", metavar='<word_count>', help='enables summary mode, and sets threshold')
    # option for requesting summary to console
    parser.add_argument("-e", "--echo", action='store_true', help='script output prints summary')
    # option to update all score entries based on each incremental selection
    parser.add_argument("-u", "--update", action='store_true', help='update score values per selected sentence')
    # option to view debug messages
    parser.add_argument("-v", "--verbose", action='store_true')
    # sets parser to remove stopwords
    parser.add_argument("-w", "--stopword", action='store_true', help='turns on stop word removal')
    parser.add_argument("-i", metavar='<file>', help='uses <file> as list of words to include')
    parser.add_argument("-o", metavar='<file>', help='outputs summary to <file>')
    args = parser.parse_args()
    dp = DocParse()
    if args.stem: dp.stemming_on = True
    if args.stopword: dp.stop_word_on = True
    if args.echo:
        dp.summary = True
        dp.max_words_in_summary = 0
    if args.threshold:
        dp.summary = True
        dp.use_threshold = True
        dp.max_words_in_summary = int(args.threshold)
    if args.distinct:
        dp.keep_all = False
    if args.normalize:
        dp.normalize = True
    if args.score:
        dp.score = args.score
    if args.update:
        dp.update = True
    if args.o:
        dp.outfile = args.o
    if args.run:
        if args.i:
            dp.include_file = args.i
        dp.tokenize(args.infile)
        dp.find_dominating_set(args.run)
    after = dt.datetime.now()
    if timer:
        print('before:', before)
        print('after :', after)
        print('total :', after - before)


if __name__ == "__main__":
    main()
