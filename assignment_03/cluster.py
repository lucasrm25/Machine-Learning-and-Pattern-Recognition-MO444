import csv
import logging
import math
import os
import pickle
import threading

from datetime import datetime
from collections import Counter
from nltk import ngrams

class Cluster ():

    def __init__ (self, csv_file = "news_headlines.csv"):

        logging.basicConfig(filename = 'counts.log',level=logging.DEBUG)

        self.csvReader = csv.reader (open (csv_file, "r"), delimiter = ",")

        self.wordGramsDict = {}
        self.wordGramsFeatures = {}

    def readCsv (self):

        self.dates = []
        self.headlines = []

        for i,row in enumerate(self.csvReader):

            try:
                self.dates.append (datetime.strptime (row [0], "%Y%m%d"))
                self.headlines.append (row [1])
            except:
                print ("Exception at row " + str(i))



    def computeIDF (self, element, dictionnary, listElements):

        if element not in dictionnary.keys ():
            counter = 0

            for l in listElements:
                if element in l:
                    counter = counter + 1

            dictionnary [element] = math.log (len(listElements) / counter)
        else:
            print (dictionnary [element])

        return dictionnary [element]

    def preprocessTerms (self):

        print ("Computing terms")

        if os.path.isfile ("counts/termDict.pkl"):
            self.termDict = pickle.load (open("counts/termDict.pkl", "rb"))
        else:

            self.termDict = {}

            for i, headline in enumerate(self.headlines):

                alreadyList = []

                # Compute terms IDF
                terms = headline.split()
                for term in terms:
                    if term not in self.termDict.keys ():
                        self.termDict [term] = 1
                    elif term not in alreadyList:
                        self.termDict [term] = self.termDict [term] + 1

                    alreadyList.append (term)

                logging.info ("Term " + str(i) + "th computed")

            pickle.dump (self.termDict, open("counts/termDict.pkl", "wb"))

        if os.path.isfile ("features/termFeatures.pkl"):
            self.termFeatures = pickle.load (open("features/termFeatures.pkl", "rb"))
        else:
            self.termFeatures = []

            N = len (self.headlines)

            for i, headline in enumerate(self.headlines):

                # Compute TF-IDF for each term in headline
                terms = headline.split()
                termsSize = len(terms)
                counter = Counter (terms)
                termsList = []

                for (term, count) in counter.most_common ():
                    termsList.append ((term, count / termsSize * math.log (N / self.termDict [term])))

                self.termFeatures.append (termsList)

                logging.info ("FEATURE Term " + str(i) + "th computed")

            pickle.dump (self.termFeatures, open("features/termFeatures.pkl", "wb"))

    def preprocessChar4Grams (self):

        print ("Computing char 4-grams")

        if os.path.isfile ("counts/char4GramsDict.pkl"):
            self.char4gramsDict = pickle.load (open("counts/char4GramsDict.pkl", "rb"))
        else:

            self.char4gramsDict = {}

            for i, headline in enumerate(self.headlines):

                alreadyList = []

                # Compute all char level 4-grams
                for gram in ngrams (" " + headline + " ", 4):
                    if gram not in self.char4gramsDict.keys ():
                        self.char4gramsDict [gram] = 1
                    elif gram not in alreadyList:
                        self.char4gramsDict [gram] = self.char4gramsDict [gram] + 1

                    alreadyList.append (gram)

                logging.info ("Char 4-gram " + str(i) + "th computed")

            pickle.dump (self.char4gramsDict, open("counts/char4GramsDict.pkl", "wb"))

        if os.path.isfile ("features/char4GramsFeatures.pkl"):
            self.char4GramsFeatures = pickle.load (open("features/char4GramsFeatures.pkl", "rb"))
        else:

            N = len (self.headlines)

            self.char4GramsFeatures = []

            for i, headline in enumerate(self.headlines):

                # Compute TF-IDF for each term in headline
                grams = ngrams (" " + headline + " ", 4)
                gramsSize = len(list(grams))
                counter = Counter (ngrams (" " + headline + " ", 4))
                gramsList = []

                for (gram, count) in counter.most_common ():
                    gramsList.append ((gram, count / gramsSize * math.log (N / self.char4gramsDict [gram])))

                self.char4GramsFeatures.append (gramsList)

                logging.info ("FEATURE Char 4-gram " + str(i) + "th computed")

            pickle.dump (self.char4GramsFeatures, open("features/char4GramsFeatures.pkl", "wb"))

            print ("tchau")

    def preprocessWordNGrams (self, n = 3):

        print ("Computing word " + str(n) + " grams")

        if os.path.isfile ("counts/word" + str(n) + "gramsDict.pkl"):
            self.wordGramsDict[n] = pickle.load (open("counts/word" + str(n) + "gramsDict.pkl", "rb"))
        else:
            self.wordGramsDict[n] = {}

            for i, headline in enumerate(self.headlines):

                alreadyList = []

                # Compute all word level n-grams
                for gram in ngrams (("BEG " + headline + " END").split (), n):
                    if gram not in self.wordGramsDict[n].keys ():
                        self.wordGramsDict[n][gram] = 1
                    elif gram not in alreadyList:
                        self.wordGramsDict[n][gram] = self.wordGramsDict[n][gram] + 1

                    alreadyList.append (gram)

                logging.info ("Word " + str(n) + "-gram " + str(i) + "th computed")

            pickle.dump (self.wordGramsDict[n], open("counts/word" + str(n) + "gramsDict.pkl", "wb"))

        if os.path.isfile ("features/word" + str(n) + "gramsFeatures.pkl"):
            self.wordGramsFeatures[n] = pickle.load (open("features/word" + str(n) + "gramsFeatures.pkl", "rb"))
        else:
            self.wordGramsFeatures[n] = []

            N = len (self.headlines)

            for i, headline in enumerate(self.headlines):

                # Compute TF-IDF for each term in headline
                grams = ngrams (("BEG " + headline + " END").split (), n)
                gramsSize = len(list(grams))
                counter = Counter (ngrams (("BEG " + headline + " END").split (), n))
                gramsList = []

                for (gram, count) in counter.most_common ():
                    gramsList.append ((gram, count / gramsSize * math.log (N / self.wordGramsDict[n][gram])))

                self.wordGramsFeatures[n].append (gramsList)

                logging.info ("FEATURE Word " + str(n) + "-gram " + str(i) + "th computed")

            pickle.dump (self.wordGramsFeatures[n], open("features/word" + str(n) + "gramsFeatures.pkl", "wb"))

            print ("tchau")

    def preprocess (self):

        termThread = threading.Thread (target = self.preprocessTerms)
        char4gramThread = threading.Thread (target = self.preprocessChar4Grams)
        word2GramThread = threading.Thread (target = self.preprocessWordNGrams, args = (2,))
        word3GramThread = threading.Thread (target = self.preprocessWordNGrams, args = (3,))

        termThread.start ()
        char4gramThread.start ()
        word2GramThread.start ()
        word3GramThread.start ()

        termThread.join ()
        char4gramThread.join ()
        word2GramThread.join ()
        word3GramThread.join ()

    def computeFeatures (self):
        print (self.headlines[19347])
        print (self.termDict ["beckham"])
        print (self.termDict ["charity"])
        print (self.termFeatures[19347])
        print (self.headlines[999995])
        print (self.char4gramsDict[(' ', 'e', 'l', 'o')])
        print (self.char4GramsFeatures[999995])
        print (self.headlines[569])
        print (self.wordGramsFeatures[2][569])
        print (self.headlines[31232])
        print (self.wordGramsFeatures[3][31232])
