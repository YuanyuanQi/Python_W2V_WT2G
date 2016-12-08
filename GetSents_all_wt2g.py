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
from nltk.tokenize import word_tokenize
import sys, re, os
import pandas as pd
from pandas import Series, DataFrame
import random
#import cPickle
reload(sys) #python2.7
#from imp import reload#python3.0
# this code used to generate a dict which instore the documents by sentence and sentence from the big dict key is docno and value is the whole text

def saveObjToFile(FileName,obj):
	fw = open(FileName,"wb")
	pickle.dump(obj,fw,protocol=2)
	fw.close()


def get_S(stopwords_file):
	List = []
	SFile = open(stopwords_file,"rb")
	Lines = SFile.readlines()
	#print Lines
	for line in Lines:
		#print line
		#print line.strip()
		List.append(line.strip())
	return List

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
	for word in vocab:
		if word not in word_vecs and vocab[word] >= min_df:
			word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
	string = re.sub(r"\'s", " \'s", string) 
	string = re.sub(r"\'ve", " \'ve", string) 
	string = re.sub(r"n\'t", " n\'t", string) 
	string = re.sub(r"\'re", " \'re", string) 
	string = re.sub(r"\'d", " \'d", string) 
	string = re.sub(r"\'ll", " \'ll", string) 
	#string = re.sub(r",", " , ", string) 
	string = re.sub(r"!", " ! ", string) 
	string = re.sub(r"\(", " \( ", string) 
	string = re.sub(r"\)", " \) ", string) 
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\*", " \* ", string) 
	string = re.sub(r"\s{2,}", " ", string)    
	return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
	string = re.sub(r"[^A-Za-z0-9().,!?\'\`]", " ", string)   
	string = re.sub(r"\s{2,}", " ", string)    
	return string.strip().lower()


def exer(clean_string=True):
	stopwords_file = "/home/echo/Documents/word2vec/CNN_keras_demo/stopword-list.txt"
	SList = get_S(stopwords_file)
	filepath = "/home/echo/Documents/WT2G/WT01/DOCs_TEXT_Dict"
	Dict = pickle.load(open(filepath,"rb"))
	count = 0
	vocab = defaultdict(float)
	for key,value in Dict.items():
		count =  count +1
		if (count<3):
			#print key
			cnt = 0
			for line ,sent in value.items():
				rev = []
				rev.append(sent.strip())
				if clean_string:
					orig_rev = clean_str(" ".join(rev))
				else:
					orig_rev = " ".join(rev).lower()
				orig_rev_list= orig_rev.split()
				new_list = [w for w in orig_rev_list if not w in SList]
				orig_rev_new =" ".join(new_list)
				words = set(orig_rev_new.split())
				for word in words:
					vocab[word] += 1
				datum  = {"y":1,"docname":key,"line_num":line, "text": orig_rev_new, "num_words": len(orig_rev_new.split())}

	
	
	
	
	
if __name__ == "__main__":
	Filepath = "/home/echo/Documents/word2vec/CNN_keras_demo/query_docs/"
	'''DocsDict= pickle.load(open("/home/echo/Documents/word2vec/CNN_keras_demo/DocsDict","rb"))
	#key is target docnos(WT02-B33-20_1) not whole docno of corpus ,value is dict (key is word,value is tf og word)
	DocsLen= pickle.load(open("/home/echo/Documents/word2vec/CNN_keras_demo/DocsLen","rb"))
	#key is docno:"WT01-B11-196",value is the length of docno without stopwords and repeated words but not stemmed
	DocFrqDict = pickle.load(open("/home/echo/Documents/WT2G/Sentence/WordTDF_Dict","rb"))
	#key is word,value is number of docments which words appeared
	numberOfDocuments = len(DocsLen)
	averageDocumentLength = sum(DocsLen.values())/len(DocsLen)#'''
	#Score_Window(FilePath,DocsDict,DocsLen,DocFrqDict,numberOfDocuments,averageDocumentLength)
	#exer()
	#Score_WindowN(Filepath)
	exer(clean_string=True)
	
