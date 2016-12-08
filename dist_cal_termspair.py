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
#this code is for calculating the dist of terms pair from documents and query 

def saveObjToFile(FileName,obj):
	fw = open(FileName,"wb")
	pickle.dump(obj,fw,protocol=2)
	fw.close()

def WriteFile():
	Idfile = open('List_all','r')
	List = pickle.load(Idfile)
	csvfile = file('wt10gResult.csv', 'wb')
	writer = csv.writer(csvfile,delimiter=',')
	Sig = False
	List_key = []
	for lists in List:
		#print lists
		for tuples in lists:
			if Sig == False:
				List_key.append(tuples[0])
		Sig = True
	writer.writerow(List_key)
	for lists in List:	
		List_value = []
		for tuples in lists:
			List_value.append(tuples[1])
		writer.writerow(List_value)	
	csvfile.close()

def vectors_Cos(v1,v2):
	x = np.array(v1)
	y = np.array(v2)
	Lx = np.sqrt(x.dot(x))
	Ly = np.sqrt(y.dot(y))
	cos_angle =(x.dot(y))/(Lx*Ly)
	return cos_angle
	QDs_List= []
	QDict = {}
	for word in query:
		#print word
		word_index = word_idx_map[word]
		Q_w2v = W[word_index]
		EDSList =[]
		for sent_index in sent:
			if sent_index != 0:
				S_w2v =W[sent_index]
				EDSList.append(vectors_Euclidean(Q_w2v,S_w2v))
		QDict[word] = EDSList
		QDs_List.append(EDSList)
	return QDs_List,QDict

def CDistance(sent,query,W,word_idx_map):
	QDs_List= []
	QDict = {}
	QWDict ={}
	QDsIndex = {}
	for word_w in query:
		word = word_w.strip(',').lower()
		word_index = word_idx_map[word]
		Q_w2v = W[word_index]
		#print Q_w2v
		CDSList =[]
		SWIndex = []
		for sent_index in sent:
			if sent_index != 0:
				S_w2v =W[sent_index]
				#print S_w2v
				CDSList.append(vectors_Cos(Q_w2v,S_w2v))
				#print CDSList
				SWIndex.append(sent_index)
			'''else:
				S_w2v =W[sent_index]
				#print S_w2v
				CDSList.append(0)
				SWIndex.append(sent_index)'''
		#if len(CDSList)==0:
		#	print sent
		QDict[word] = CDSList
		QDs_List.append(CDSList)
		QDsIndex [word] =SWIndex
	#print sent[CDList.index(max(CDList))]
	return QDs_List,QDict,QDsIndex



if __name__ == "__main__":
	#print "Load Files:w2v,Docs_Dict,f_idx_map(w2v_index_qi)"
	#vocab :key is word string ,value is term frequency of word in whole corpus
	#w2v : key is word ,value is vector of word
	# W: key is index,value is vector
	# word_idx_map:key is word,value is index
	# Docs_Dict :key is docno ,value is datum(datum  = {"y":1,"docname":docno,"line_num":line, "text": orig_rev_new, "num_words": len(orig_rev_new.split())})
	#w2v = pickle.load(open("/home/echo/Documents/word2vec/CNN_keras_demo/query_all/w2v","rb"))
	#print "Loaded w2v File"
	#with open("/home/echo/Documents/word2vec/CNN_keras_demo/query_all/DocsDict", "r") as f:
    		#DocsDict = pickle.load(f)
	#DocsDict= pickle.load(open("/home/echo/Documents/word2vec/CNN_keras_demo/query_all/DocsDict","rb"))
	#print "Loaded Docsdict"
	print "Start to Calculate Distance of terms pair from document and query"
	QueryFilesPath = "/home/echo/Documents/word2vec/CNN_keras_demo/query_docs"
	QueryFiles = os.listdir(QueryFilesPath)
	for queryfile in QueryFiles:
		queryterms = re.split(r'[, ]',queryfile)
		for item in queryterms:
			if item =="":
				queryterms.remove(item)
				continue
		
