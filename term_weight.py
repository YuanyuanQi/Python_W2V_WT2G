#!/usr/bin/python
# -*- coding: UTF-8 -*-
import gzip  
import math
import sys
import time
import datetime
import string
import re
import operator
import pickle
import os
import math
import csv
import numpy as np
import cPickle
from collections import defaultdict
import sys, re, os
import pandas as pd
import random
#import cPickle
reload(sys) #python2.7
#from imp import reload#python3.0
# now we are using Dict files which written in python 3 with protocol = 2
# @author echo
# this code is to analysis the query term's weight
#
def Analysis(filepath):
	CsvFiles = os.listdir(filepath)
	queryterms = filepath.split("/")[len(filepath.split("/"))-2].strip().split(" ")
	query= []
	for term in queryterms:
		query.append(term.strip(","))
	for csvf in CsvFiles:
		if csvf.find("new") == -1:
			continue
		else:
			csvfpath = os.path.join(filepath,csvf)
			count = 0
			with open(csvfpath,'rb') as f:
				count  = count +1
				f_csv = csv.reader(f)
				headers = next(f_csv)
				#print headers[1:len(headers)-1]
				for row in f_csv:
						
			'''docno_label = csvf.replace(".csv","")#docno_label = WT13-B23-171_1
			docno = docno_label.split("_")[0]
			docLength = DocsLen[docno]
			if DocsDict.has_key(docno_label):
				vocab = DocsDict[docno_label]
				for term in query:
					if vocab.has_key(term):
						tf = vocab[term]
					else:
						tf = 0
					if DocFrqDict.has_key(term):
						documentFrequency= DocFrqDict[term]
					else:
						documentFrequency = 0#'''
					
			
def TF_IDF(term):
	return 0	
		
		
		
if __name__ == "__main__":
	'''stopwords_file = "/home/echo/Documents/word2vec/CNN_keras_demo/stopword-list.txt"
	SList = get_S(stopwords_file)
	DocsDict= pickle.load(open("/home/echo/Documents/word2vec/CNN_keras_demo/DocsDict","rb"))
	#key is target docnos(WT02-B33-20_1) not whole docno of corpus ,value is dict (key is word,value is tf og word)
	DocsLen= pickle.load(open("/home/echo/Documents/word2vec/CNN_keras_demo/DocsLen","rb"))
	#key is docno:"WT01-B11-196",value is the length of docno without stopwords and repeated words but not stemmed
	DocFrqDict = pickle.load(open("/home/echo/Documents/WT2G/Sentence/WordTDF_Dict","rb"))
	#key is word,value is number of docments which words appeared#'''
	target = "/home/echo/Documents/word2vec/CNN_keras_demo/query_docs/women clergy/"
	Analysis(target)
