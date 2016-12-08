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
#import pandas as pd
import random
#import cPickle
reload(sys) #python2.7
#from imp import reload#python3.0
# now we are using Dict files which written in python 3 with protocol = 2
# this code use to genrate index for all documents in WT2G corpus 
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

def get_vocab(data_all,SList,clean_string=True):
	vocab = defaultdict(float)
	Files = os.listdir(data_all)
	filepathlist =[]
	for File in Files:
		if File.find("WT") != -1:
			Filepath = os.path.join(data_all,File)
			if os.path.exists(Filepath):
				filepath = os.path.join(Filepath,"DOCs_TEXT_Dict")
				foutpath = filepath.replace("_TEXT_","datum")
				filepathlist.append(foutpath)
				Docs_Dict = pickle.load(open(filepath,"rb"))
				print "Processing at: " +File
				#line_num = 0
				DocsdatumDict ={}
				for docno,text in Docs_Dict.items():
					revs = []
					for line,sent in text.items():
						rev =[]
						if type(sent) == int:
							continue
						else:
							rev.append(str(sent.encode('utf-8','ignore').strip()))
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
					#docsdict[docno] = revs    #key is document name ,value is the list which contanins all sentences information of the document
					DocsdatumDict[docno]= revs
				saveObjToFile(foutpath,DocsdatumDict)
	print "All vocab is : " + str(len(vocab))
	saveObjToFile("Docsdatumdict_filepaths",filepathlist)
	return vocab,filepathlist

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


def Get_QueryDict():
	with open("/home/echo/Documents/word2vec/CNN_keras_demo/topics.wt2g","rb") as f:
		File_all = f.read()
		Buck_List = File_all.split('<top>')
		Dict ={}
		for item in Buck_List:
			item = item.strip('\n')
			Num_Begin = item.find('<num>')+5
			Num_End =  item.find('<title>')
			num = ""
			num_q =""
			if Num_Begin > 0 and Num_End >0:
				num = item[Num_Begin:Num_End].strip('\n')
				num_q = num.split(':')[1].strip(' ')
			title =""
			newtitle =[]
			Title_Begin = item.find('<title>')+7
			Title_end = item.find('<desc>')
			if Title_Begin>0 and Title_end>0:
				title = item[Title_Begin:Title_end].strip('\n')
				title = title.replace(",","")
				newtitle = title.split()
			if len(newtitle) >0:
				Dict[int(num_q)]= newtitle
		items = Dict.items()
		items.sort()
		return items	

#this part we calculate the query“women clergy”
if __name__ == "__main__":
	data_all = "/home/echo/Documents/WT2G/"
	print "Star Processing"
	stopwords_file = "/home/echo/Documents/word2vec/CNN_keras_demo/stopword-list.txt"
	SList = get_S(stopwords_file)
	print "loading vocab data..."
	#vocab,DocsDict,revfs = get_vocab(data_all,SList,clean_string=True)
	vocab,filepathlist= get_vocab(data_all,SList,clean_string=True)
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
	#np.savez('/home/irlab/Documents/coding/corpus_all/embedding_weights_QI.npz',embedding_weights=W)
	#what the npz file use for?
	del vocab
	del w2v
	#queryDict = Get_QueryDict()
	#print queryDict #key is query id :450 .key is list which contain all terms in query
	print "Calculating and Writing CSV file"
	#num = len(queryDict)
	#count = 0
	query_terms ="women clergy"
	query = ["women","clergy"]
	Docsddict={}
	print "Processing : "+ query_terms
	TargetFiles = os.listdir(data_all)
	for dictpath in filepathlist:
		print "Processing :　"+dictpath.split("/")[len(dictpath.split("/"))-2]
		docsdict = pickle.load(open(dictpath,"rb"))
		for docno,revs in docsdict.items():
			fout = "/home/echo/Documents/word2vec/CNN_keras_demo/corpus_all/"+query_terms+"/"+docno+"_dist.csv"
			if os.path.isfile(fout):
				continue
			else:
				fo_dist = open(fout,"wb")
				writer = csv.writer(fo_dist)
				Header=["line_num"]
				for term in query:
					Header.append(term+"_dist")
					Header.append(term+"_word")
				Header.append("sum_dist_terms")
				writer.writerow(Header)
				DocCDict = {}
				newlist =sorted(revs, key=lambda datum: datum["line_num"])#for read the line from begining to end
				doc ={}
				for rev in newlist:#rev is one sentence in document //rev is list 
						#sent = get_idx_from_sent(rev["text"], word_idx_map, max_l=max_len, k=300, filter_h=5)
					sent = new_get_idx_from_sent(rev["text"], word_idx_map)
					if sum(sent) > 0:#kick off the sentence full of stopwords
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
				Docsddict[docno]=doc
				fo_dist.close()
			saveObjToFile("/home/echo/Documents/word2vec/CNN_keras_demo/"+query_terms+"_DocsdistDict",Docsddict)	
	print "done"#'''





