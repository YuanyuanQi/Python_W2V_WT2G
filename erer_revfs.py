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
#sent is  rev["text"] x is list for vector for each word in sent.
def new_get_idx_from_sent(sent, word_idx_map):
	x = []
	words = sent.split()
	for word in words:
		if word in word_idx_map:
			x.append(word_idx_map[word])
	return x

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
	for word in vocab:
		if word not in word_vecs and vocab[word] >= min_df:
			word_vecs[word] = np.random.uniform(-0.25,0.25,k) 

def vectors_Cos(v1,v2):
	x = np.array(v1)
	y = np.array(v2)
	Lx = np.sqrt(x.dot(x))
	Ly = np.sqrt(y.dot(y))
	cos_angle =(x.dot(y))/(Lx*Ly)
	return cos_angle

def new_CDistance(sent,term,W,word_idx_map):
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

def saveCsvFile(FileName,Dict):
	fo_score = open(FileName,"wb")
	writer = csv.writer(fo_score)
	Header = ["docnolabel","score"]
	writer.writerow(Header)
	items = sorted(Dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
	for key,value in items:
		List =[]
		List.append(key)
		List.append(value)
		writer.writerow(List)
	fo_score.close()

def GetNP(sentence):
	List = []
	for word in sentence:
		word_index = word_idx_map[word]
		word_w2v = W[word_index]
		List.append(word_w2v)
	return np.matrix(List)
		
		
if __name__ == "__main__":
	if os.path.isfile("/home/echo/Documents/word2vec/CNN_keras_demo/query_all/vocab"):
		print "Loading Vocab and docsdict"
		vocab = pickle.load(open("/home/echo/Documents/word2vec/CNN_keras_demo/query_all/vocab","rb"))
		docsdict = pickle.load(open("/home/echo/Documents/word2vec/CNN_keras_demo/query_all/docsdict","rb"))
	else:
		print "Generating Vocab and docsdict"
		stopwords_file = "/home/echo/Documents/word2vec/CNN_keras_demo/stopword-list.txt"
		SList = get_S(stopwords_file)
		clean_string= True
		docsdict ={}
		filepath = "/home/echo/Documents/WT2G/WT04/DOCs_TEXT_Dict"
		Docs_Dict = pickle.load(open(filepath,"rb"))
		vocab = defaultdict(float)
		for docno,Dict in Docs_Dict.items():
			revs = []
			for line,value in Dict.items():
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
					vocab[word] += 1
				datum  = {"docname":docno,"line_num":int(line), "text": orig_rev_new, "num_words": len(orig_rev_new.split())}
				revs.append(datum)#store all sentences information in one file
				docsdict[docno] = revs    #key is document name ,value is the list which contanins all sentences information of the document
		print "Got Vocab and docsdict"
		saveObjToFile("/home/echo/Documents/word2vec/CNN_keras_demo/query_all/vocab",vocab)
		saveObjToFile("/home/echo/Documents/word2vec/CNN_keras_demo/query_all/docsdict",docsdict)
	w2v_file = "/home/echo/Documents/word2vec/CNN_keras_demo/GoogleNews-vectors-negative300.bin"
	print "loading word_vector BIN file "
	w2v = load_bin_vec(w2v_file,vocab)
	print ("loaded")
	print "num words of vocab already in word2vec: " + str(len(w2v))
	add_unknown_words(w2v, vocab)
	W, word_idx_map = get_W(w2v)
	print "Calculating and Writing CSV file"
	#query=["women","clergy"]
	query = ["women","clergy"]	
	query_terms = "women clergy"
	Docsdict={}
	count = 0
	for docno,revs in docsdict.items():
		count = count +1
		if count <2:
			fo_dist = open("/home/echo/Documents/word2vec/CNN_keras_demo/query_all/"+query_terms+"/"+docno+"_dist.csv","wb")
			writer = csv.writer(fo_dist)
			Header=["line_num"]
			for term in query:
				Header.append(term+"_dist")
				#Header.append(term+"_word")
			Header.append("sum_dist_terms")
			writer.writerow(Header)
			DocCDict = {}
			newlist =sorted(revs, key=lambda datum: datum["line_num"])
			doc ={}
			query_np = GetNP(query) 
			#my_matrix = np.loadtxt(open("c:\\1.csv","rb"),delimiter=",",skiprows=0)  
			for rev in newlist:#rev is one sentence in document //rev is list 
				#sent = get_idx_from_sent(rev["text"], word_idx_map)
				List=[]
				List.append(rev["line_num"])
				sent_np = GetNP(rev["text"].split())
				#temp = sent_np.dot(query_np.T)
				temp = sent_np.dot(query_np.T)/np.outer(np.linalg.norm(sent_np, axis=1),np.linalg.norm(query_np, axis=1))
				#temp = (sent_np/np.linalg.norm(sent_np,axis=1)).dot((query_np/np.linalg.norm(query_np,axis=1)).T)
				score = np.max(temp, axis = 0)# axis=0 means max down lower the rows, which is the column some
				
				#print score
				sums = np.sum(score,axis = 1)#axis=1 means sum right over the columns, which is the row sum
				for item in score.tolist():
					#print item
					List.extend(item)
				List.append(sum(List[1:]))
				#for item in sums.tolist():
					#List.extend(item)
				print List
				print "ORIGIN:"
				sent = new_get_idx_from_sent(rev["text"], word_idx_map)
				if sum(sent) > 0:#kick off the sentence full of stopwords
							#Cosine Distance
					List1=[]
					List1.append(rev["line_num"])
						#termsdist =[]
					for term in query:
						CDSList,SWIndex = new_CDistance(sent,term,W,word_idx_map)
								#CDSList : the dist between all words in one sentence and one query term
								#SWIndex : nstore all words index(word_index map) in one sentence 
						lindex = CDSList.index(max(CDSList))#get the max dist of word from one sentence and local the index in the list
								#sIndex = QDsIndex[key]#the sentence words index by the term of query
						windex = SWIndex[lindex]#from the index of list to locate the word index in whole word——map
						word = word_idx_map.keys()[word_idx_map.values().index(windex)]
								#datum = {"docno":docno,"line_num":rev["line_num"],"key":term,"word":word,"dist":max(value)}
								#doc.append(datum)#instore one document with all sentences dists with all terms in query
							#termsdist.append(max(CDSList))
						List1.append(max(CDSList))
						#List1.append(word)
					List1.append(sum(List1[1:]))
					print List1
				#np.savetxt('new.csv', my_matrix, delimiter = ',')  
				'''if sum(sent) > 0:#kick off the sentence full of stopwords
					#Cosine Distance
					List=[]
					List.append(rev["line_num"])
					termsdist =[]
					for term in query:
						CDSList,SWIndex = new_CDistance(sent,term,W,word_idx_map)
						#CDSList : the dist between all words in one sentence and one query term
						#SWIndex : nstore all words index(word_index map) in one sentence 
						lindex = CDSList.index(max(CDSList))#get the max dist of word from one sentence and local the index in the list
						#sIndex = QDsIndex[key]#the sentence words index by the term of query
						windex = SWIndex[lindex]#from the index of list to locate the word index in whole word——map
						word = word_idx_map.keys()[word_idx_map.values().index(windex)]
						#datum = {"docno":docno,"line_num":rev["line_num"],"key":term,"word":word,"dist":max(value)}
						#doc.append(datum)#instore one document with all sentences dists with all terms in query
						termsdist.append(max(CDSList))
						List.append(max(CDSList))
						List.append(word)
					List.append(sum(termsdist))
					termsdist =[]
					doc[rev["line_num"]]=List
					writer.writerow(List)
			Docsdict[docno]=doc
			fo_dist.close()
			#Keys[term] =Docsdict
		saveObjToFile("/home/echo/Documents/word2vec/CNN_keras_demo/query_all/"+query_terms+"_DocsdistDict",Docsdict)	
		print "done"#'''
				writer.writerow(List)
			fo_dist.close()
	
		
