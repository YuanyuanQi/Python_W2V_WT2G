#!/usr/bin/python
# -*- coding: UTF-8 -*-
#this code is to classify the query relative documents from POS_1/NEG_1 to generate dict which key is the docno_label+linenum and value is [key+word,max(dist from words in sentence with query term)]
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
# this code use to genrate a dict which instore the dict by query from the target file data_folder
def saveObjToFile(FileName,obj):
	fw = open(FileName,"wb")
	pickle.dump(obj,fw,protocol=2)
	fw.close()

def load_bin_vec(fname,vocab):
	"""
	Loads 300x1 word vecs from Google (Mikolov) word2vec
	"""
	word_vecs = {}
	with open(fname, "rb") as f:
		header = f.readline()
		vocab_size, layer1_size = map(int, header.split())
		binary_len = np.dtype('float32').itemsize * layer1_size
		for line in xrange(vocab_size):
			word = []
			while True:
				ch = f.read(1)
				if ch == ' ':
					word = ''.join(word)
					break
				if ch != '\n':
					word.append(ch)   
			if word in vocab:
				word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
			else:
				f.read(binary_len)
	return word_vecs

def clean_str(string, TREC=False):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
	string = re.sub(r"\'s", " \'s", string) 
	string = re.sub(r"\'ve", " \'ve", string) 
	string = re.sub(r"n\'t", " n\'t", string) 
	string = re.sub(r"\'re", " \'re", string) 
	string = re.sub(r"\'d", " \'d", string) 
	string = re.sub(r"\'ll", " \'ll", string) 
	string = re.sub(r",", " , ", string) 
	string = re.sub(r"!", " ! ", string) 
	string = re.sub(r"\(", " \( ", string) 
	string = re.sub(r"\)", " \) ", string) 
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\*", " \* ", string) 
	string = re.sub(r"\s{2,}", " ", string)    
	return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
	string = re.sub(r"\s{2,}", " ", string)    
	return string.strip().lower()

def get_W(word_vecs, k=300):
	"""
	Get word matrix. W[i] is the vector for word indexed by i
	"""
	vocab_size = len(word_vecs)
	word_idx_map = dict()
	W = np.zeros(shape=(vocab_size+1, k))            
	W[0] = np.zeros(k)
	i = 1
	for word in word_vecs:
		W[i] = word_vecs[word]
		word_idx_map[word] = i
		i += 1
	return W, word_idx_map

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

#sent is  rev["text"] x is list for vector for each word in sent
def get_idx_from_sent(sent, word_idx_map, max_l=795, k=300, filter_h=5):
	x = []
	pad = filter_h - 1
	for i in xrange(pad):
		x.append(0)
	words = sent.split()
	for word in words:
		if word in word_idx_map:
			x.append(word_idx_map[word])
	while len(x) < max_l+2*pad:
		x.append(0)
	#print "words in line"
	#print sent
	#print "output is list to store words which  matched with vector? "
	#print x
	return x

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
	for word in vocab:
		if word not in word_vecs and vocab[word] >= min_df:
			word_vecs[word] = np.random.uniform(-0.25,0.25,k) 

def get_vocab(data_folder,SList,clean_string=True):
	vocab = defaultdict(float)
	docdict = {}
	revfs = []
	POS_DFiles = os.listdir(data_folder[0])
	NEG_DFiles = os.listdir(data_folder[1])
	for POS in POS_DFiles:
		revs = []
		filepath = os.path.join(data_folder[0],POS)
		Doc_Dict = pickle.load(open(filepath,"rb"))
		#key is line_num value is sentence 
		line_num = 0
		docno = POS.split("_")[1]
		for line,value in Doc_Dict.items():
			line_num = line_num +1
			rev = []
			#print value
			if type(value) == int:
				continue
			else:
				rev.append(str(value.encode('utf-8','ignore').strip()))
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
			#datum  = {"y":1,"docname":NEG,"line_num":line_num, "text": orig_rev, "num_words": len(orig_rev.split())}
			datum  = {"y":1,"docname":docno,"line_num":int(line), "text": orig_rev_new, "num_words": len(orig_rev_new.split())}
			#print datum
			revs.append(datum)#store all sentences information in one file
		docdict[POS] = revs    #key is document name ,value is the list which contanins all sentences information of the document
		revfs.append(docdict)#store all files information 
		#print ("POS files that have words in vocab:   "+str(len(vocab)))'''
	for NEG in NEG_DFiles:
		docno = NEG.split("_")[1]
		revs = []
		filepath = os.path.join(data_folder[1],NEG)
		Doc_Dict = pickle.load(open(filepath,"rb"))
		line_num = 0
		for line,value in Doc_Dict.items():
			line_num = line_num +1
			rev = []
			if type(value) == int:
				continue
			else:
				rev.append(str(value.encode('utf-8','ignore').strip()))
			if clean_string:
				orig_rev = clean_str(" ".join(rev))
			else:
				orig_rev = " ".join(rev).lower()
			orig_rev_list= orig_rev.split()
			new_list = [w for w in orig_rev_list if not w in SList]
			orig_rev_new =" ".join(new_list)
			words = set(orig_rev_new.split())
			for word in words:
				if word in SList:
					print word + "  is in the vocab"
				vocab[word] += 1
			#datum  = {"y":1,"docname":NEG,"line_num":line_num, "text": orig_rev, "num_words": len(orig_rev.split())}
			datum  = {"y":0,"docname":docno,"line_num":int(line), "text": orig_rev_new, "num_words": len(orig_rev_new.split())}
			#print datum
			revs.append(datum)#store all sentences information in one file
		docdict[NEG] = revs    #key is document name ,value is the list which contanins all sentences information of the document
		revfs.append(docdict)#store all files information '''
	print "ALL vocab:   "+str(len(vocab))
	#print  revfs
	return vocab,docdict,revfs

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
		QDict[word] = CDSList
		QDs_List.append(CDSList)
		QDsIndex [word] =SWIndex
	#print sent[CDList.index(max(CDList))]
	return QDs_List,QDict,QDsIndex


def newCDistance(sent,term,W,word_idx_map):
	word = term.strip(',').lower()
	word_index = word_idx_map[word]
	Q_w2v = W[word_index]
	CDSList =[]
	SWIndex = []
	for sent_index in sent:
		if sent_index != 0:
			S_w2v =W[sent_index]
				#print S_w2v
			CDSList.append(vectors_Cos(Q_w2v,S_w2v))
				#print CDSList
			SWIndex.append(sent_index)
	return CDSList,SWIndex

def Calculate_dist(querytermslist,DocsDict,max_len):
	Keys ={}
	for term in querytermslist:
		Docsdict={}
		for docno,revs in DocsDict.items():#DocsDict key is docno_label(p/n) value is list which each element is dict:datum and list is for all lines in document
			DocCDict = {}
			newlist =sorted(revs, key=lambda datum: datum["line_num"])
			doc =[]
			for rev in newlist:#rev is one sentence in document //rev is list 
				sent = get_idx_from_sent(rev["text"], word_idx_map, max_l=max_len, k=300, filter_h=5)
				if sum(sent) > 0:#kick off the sentence full of stopwords
					#Cosine Distance
					CDSList,SWIndex = CDistance(sent,querytermslist,W,word_idx_map)
					#CDSList : the dist between all words in one sentence and one query term
					#SWIndex : nstore all words index(word_index map) in one sentence 
					lindex = value.index(max(CDSList))#get the max dist of word from one sentence and local the index in the list
					sIndex = QDsIndex[key]#the sentence words index by the term of query
					windex = sIndex[lindex]#from the index of list to locate the word index in whole word——map
					word = word_idx_map.keys()[word_idx_map.values().index(windex)]
					datum = {"docno":docno,"line_num":rev["line_num"],"key":key,"word":word,"dist":max(value)}
					doc.append(datum)#instore one document with all sentences dists with all terms in query
			Docsdict [docno]=doc
		Keys[term] =Docsdict
		

if __name__ == "__main__":
	data_folder = ["/home/echo/Documents/WT2G/Sentence/POS_new","/home/echo/Documents/WT2G/Sentence/NEG_new"]
	print "Star Processing"
	stopwords_file = "/home/echo/Documents/word2vec/CNN_keras_demo/stopword-list.txt"
	SList = get_S(stopwords_file)
	print "Loading vocab data..."
	vocab,DocsDict,revfs = get_vocab(data_folder,SList,clean_string=True)
	print "Loaded"
	Doc_Words_Dict = {}
	numwords_list = []
	#print pd.DataFrame(revfs)
	for revf in revfs:
		#print revf
		for doc,sentlist in revf.items():
			doc_list = []
			for sent in sentlist:
				numwords_list.append(int(sent["num_words"]))
				doc_list.append(int(sent["num_words"]))
			Doc_Words_Dict[doc] = doc_list
	max_len = np.max(numwords_list)
	print "Max words of Sentence is : "+str(max_len)#'''
	print "loaded!"
	#vocab saved words that appeared in the POS and NEG files and relative frequency or words which is map
	w2v_file = "/home/echo/Documents/word2vec/CNN_keras_demo/GoogleNews-vectors-negative300.bin"
	print "loading word_vector BIN file "
	w2v = load_bin_vec(w2v_file,vocab)
	print ("loaded")
	print "num words already in word2vec: " + str(len(w2v))
	add_unknown_words(w2v, vocab)
	W, word_idx_map = get_W(w2v)
	# W: key is index,value is vector
	# word_idx_map:key is word,value is index
	np.savez('query_all/embedding_weights_QI.npz',embedding_weights=W)
	#what the npz file use for?
	f_idx_map = open('query_all/w2v_index_qi','w')
	for item in word_idx_map:
		f_idx_map.write(item+'\t'+str(word_idx_map[item])+'\n')
	#f_idx_map.close()
	random.shuffle(revfs)
	print "Processing the Distance"
	#for revf in revfs:
	#print "Start to Calculate Distance of terms pair from document and query"
	QueryFilesPath = "/home/echo/Documents/word2vec/CNN_keras_demo/query_docs"
	QueryFiles = os.listdir(QueryFilesPath)
	for queryfile in QueryFiles:
		queryterms = re.split(r'[, ]',queryfile)
		for item in queryterms:
			if item =="":
				queryterms.remove(item)
				continue
		Calculate_dist(queryterms,DocsDict,max_len):
	
