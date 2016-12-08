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
		line_num = 0
		for line,value in Doc_Dict.items():
			line_num = line_num +1
			rev = []
			rev.append(line.strip())
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
				#else:
					#print word + "  is NOT in the vocab"
				vocab[word] += 1
			#datum  = {"y":1,"docname":NEG,"line_num":line_num, "text": orig_rev, "num_words": len(orig_rev.split())}
			datum  = {"y":1,"docname":POS,"line_num":line_num, "text": orig_rev_new, "num_words": len(orig_rev_new.split())}
			#print datum
			revs.append(datum)#store all sentences information in one file
		docdict[POS] = revs    #key is document name ,value is the list which contanins all sentences information of the document
		revfs.append(docdict)#store all files information 
		#print ("POS files that have words in vocab:   "+str(len(vocab)))'''
	for NEG in NEG_DFiles:
		revs = []
		filepath = os.path.join(data_folder[1],NEG)
		Doc_Dict = pickle.load(open(filepath,"rb"))
		line_num = 0
		for line,value in Doc_Dict.items():
			line_num = line_num +1
			rev = []
			rev.append(line.strip())
			if clean_string:
				orig_rev = clean_str(" ".join(rev))
			else:
				orig_rev = " ".join(rev).lower()
			orig_rev_list= orig_rev.split()
			'''for word in orig_rev_list:
				if word.strip() in SList:
					#print word
					orig_rev_list.remove(word)
			#print orig_rev_list.index("a")
			for sw in SList:
				if sw in orig_rev_list:
					print sw +"  is in the orig_rev_list"
				#else:
				#	print sw +"  is in the orig_rev_list"
			orig_rev_new =" ".join(orig_rev_list) 
			words = set(orig_rev_new.split())'''
			new_list = [w for w in orig_rev_list if not w in SList]
			orig_rev_new =" ".join(new_list)
			words = set(orig_rev_new.split())
			for word in words:
				if word in SList:
					print word + "  is in the vocab"
				vocab[word] += 1
			#datum  = {"y":1,"docname":NEG,"line_num":line_num, "text": orig_rev, "num_words": len(orig_rev.split())}
			datum  = {"y":1,"docname":POS,"line_num":line_num, "text": orig_rev_new, "num_words": len(orig_rev_new.split())}
			#print datum
			revs.append(datum)#store all sentences information in one file
		docdict[NEG] = revs    #key is document name ,value is the list which contanins all sentences information of the document
		revfs.append(docdict)#store all files information '''
	print "ALL vocab:   "+str(len(vocab))
	#print  revfs
	return vocab,docdict,revfs

def vectors_Euclidean(v1,v2):
	x = np.array(v1)
	y = np.array(v2)
	Lx = np.sqrt(x.dot(x))	
	Ly = np.sqrt(y.dot(y))
	dist = np.linalg.norm(x-y)
	return dist

def vectors_Cos(v1,v2):
	x = np.array(v1)
	y = np.array(v2)
	Lx = np.sqrt(x.dot(x))
	Ly = np.sqrt(y.dot(y))
	cos_angle =(x.dot(y))/(Lx*Ly)
	return cos_angle

def EDistance(sent,query,W,word_idx_map):
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
	for word in query:
		#print word
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
	data_folder = ["/home/echo/Documents/word2vec/CNN_keras_demo/3pos3neg/POS_1","/home/echo/Documents/word2vec/CNN_keras_demo/3pos3neg/NEG_1"]
	print "Star Processing"
	stopwords_file = "/home/echo/Documents/word2vec/CNN_keras_demo/stopword-list.txt"
	SList = get_S(stopwords_file)
	print "loading vocab data..."
	vocab,DocsDict,revfs = get_vocab(data_folder,SList,clean_string=True)
	#print len(revfs)
	#print len(DocsDict)
	'''for docname,revs in DocsDict.items():
		if docname.find("WT20-B05-239") != -1 or docname.find("WT03-B13-4") != -1:
			fw = open("/home/echo/Documents/word2vec/CNN_keras_demo/3pos3neg/"+docname.split("_")[1]+"_new.txt","w",)
			#print docname
			#fw.write("Docid is :   "+docname.split("_")[1]+"\n")
			#fw.write("Query is :   "+docname.split("_")[0]+"\n")
			#fw.write("Pos/Neg(1/0) is :   "+docname.split("_")[2]+"\n")
			Dict1 = {}
			for rev in revs:
				Dict1[int(rev["line_num"])] = rev["text"]
			itemr = Dict1.items()
			sorted(itemr)
			for line_num,text in itemr:
				#print line_num
				fw.write(text+"\n")
			fw.close()# this part is for writing down some sentences in document by offer some docname'''
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
	max_l = np.max(numwords_list)
	print "Max words of Sentence is : "+str(max_l)
	'''for revf in revfs:
		#print revf
		for doc,sentlist in revf.items():
			if doc == " women clergy _WT09-B12-83_0":
				for sent in sentlist:
					print sent['num_words']+sent['line_num']
					print sent ['text']
	#'''
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
	print ("Loaded word_vector map and index")
	
	#print ("loaded stopwords_file")
	
	np.savez('data_qpd/embedding_weights_QI.npz',embedding_weights=W)
	#what the npz file use for?
	f_idx_map = open('data_qpd/w2v_index_qi','w')
	for item in word_idx_map:
		f_idx_map.write(item+'\t'+str(word_idx_map[item])+'\n')
	random.shuffle(revfs)
	print "Processing the Distance"
	pos_s, neg_s, pos_nl, neg_nl ,pos_ed, neg_ed,pos_cd, neg_cd= [], [], [], [], [], [], [], []
	#for revf in revfs:
	DocsDict1 = {}
	
	GaoDict1 = {}# key is line num in document ,value is closest distant between query term and words in perticular sentence /line.
	GaoDict2 ={}#key is the line num ,value is closest word in line with term in query
		
	for docname,revs in DocsDict.items():#DocsDict key is docno_label(p/n) value is list which each element is dict:datum and list is for all lines in document
		query = docname.split("_")[0]
		print query
		DocEDict = {}
		DocCDict = {}
		label = 0
		ListEC = []
		for rev in revs:#rev is one sentence in document //rev is list 
			label = rev['y']
			sent = get_idx_from_sent(rev["text"], word_idx_map, max_l=795, k=300, filter_h=5)
			qrylist = query.split()
			if sum(sent) > 0:#kick off the sentence full of stopwords
				#EDistance
				EDistList,EDistDict = EDistance(sent,qrylist,W,word_idx_map)
				for key,value in EDistDict.items():
					DocEDict[key+'_'+str(rev["line_num"])]=[int(rev["num_words"]),sum(value)]
				#Cosine Distance
				CDistList,CDistDict,QDsIndex = CDistance(sent,qrylist,W,word_idx_map)
				d1,d2 ={},{}
				for key,value in CDistDict.items():
					lindex = value.index(max(value))
					sIndex = QDsIndex[key]
					windex = sIndex[lindex]
					word = word_idx_map.keys()[word_idx_map.values().index(windex)]
					if key =="clergy":
						GaoDict2[docname.split("_")[1]+"/"+docname.split("_")[2]+"_"+str(rev["line_num"])] = [key+"_"+word,max(value)]
					else:
						GaoDict1[docname.split("_")[1]+"/"+docname.split("_")[2]+"_"+str(rev["line_num"])] = [key+"_"+word,max(value)]
					#GaoWDict[docname.split("_")[1]+"/"+docname.split("_")[2]+"_"+key+"_"+str(rev["line_num"])] = word
					DocCDict[key+'_'+str(rev["line_num"])]=[int(rev["num_words"]),sum(value)]
			'''if int(rev["y"])==1:
				pos_s.append(sent)
				pos_nl.append(rev["docname"]+"_"+str(rev["line_num"]))
				pos_ed.append(EDistList)
				pos_cd.append(CDistList)
			else:
				neg_s.append(sent)
				neg_nl.append(rev["docname"]+"_"+str(rev["line_num"]))
				neg_ed.append(EDistList)
				neg_cd.append(CDistList)'''
		queryL = query.split()
		E1 = {}
		E2 = {}
		List1= []
		List2 = []
		NW1 ={}
		NW2 ={}
		#wordsnum = []
		for key,value in DocCDict.items():
			#key is the each sentence and query word distance
			#print key,value
			#wordsnum.append(key.spilt('_')[2])
			if key.split('_')[0] == "women": 
				#print "women"
				#print key,value
				NW1[str(key.split('_')[1])+"_"+str(value[0])] = value[1]
				List1.append(value[1])
				#continue
			elif key.split('_')[0] =="clergy":
				#print key,value
				NW2[str(key.split('_')[1])+"_"+str(value[0])] = value[1]
				List2.append(value[1])
		E1["women_"+str(label)] = [List1,NW1]
		ListEC.append(E1)
		E2["clergy_"+str(label)] = [List2,NW2]
		ListEC.append(E2)
		DocsDict1[docname] = ListEC
		#print len(DocsDict)
	
	saveObjToFile("GaoDict1",GaoDict1)
	saveObjToFile("GaoDict2",GaoDict2)
	#saveObjToFile("GaoWDict",GaoWDict)
	#for key,value in GaoWDict.items():
		#print key +"    :     "+ str(value)
	
	
	fw = open('/home/echo/Documents/word2vec/CNN_keras_demo/3pos3neg/Oneqry_Call.txt','w')
	for docname,ECList in DocsDict1.items():
		#print docname
		#print ECList[0]
		fw.write("Docname is :  " +docname +"\n")
		for key ,value in ECList[0].items():
			#print value
			fw.write( 'query word: '+key+'\n')
			fw.write('Number of Sents in Doc :'+str(len(value[1]))+'\n')
			fw.write("Senloc_WordsNuminSen   QS_dist:   "+'\n')
			itemd = value[1].items()
			itemd.sort()
			for numwords,SQsents in itemd:
				fw.write(str(numwords)+"        "+str(SQsents)+ '\n')
			fw.write("Q_Doc_list:   "+str(sum(value[0]))+'\n')
		for key ,value in ECList[1].items():
			fw.write( 'query word: '+key+'\n')
			fw.write('Number of Sents in Doc :'+str(len(value[1]))+'\n')
			fw.write("Senloc_WordsNuminSen   QS_dist:   "+'\n')
			fitemd = value[1].items()
			itemd.sort()
			for numwords,SQsents in itemd:
				fw.write(str(numwords)+"        "+str(SQsents)+ '\n')
			fw.write("Q_Doc_list:   "+str(sum(value[0]))+'\n')
	fw.close()#'''
print "Done"
		
'''
	pos_s = np.asarray(pos_s)
	neg_s = np.asarray(neg_s)
	pos_nl = np.asarray(pos_nl)
	neg_nl = np.asarray(neg_nl)
	pos_ed = np.asarray(pos_ed)
	neg_ed = np.asarray(neg_ed)
	pos_cd = np.asarray(pos_cd)
	neg_cd = np.asarray(neg_cd)
	#print train_y.shape
	#print test_y.shape
	print "###"
	#cc = 1
	#for item in train:
	#	if len(item) != 59:
	#		print len(item)
	#		print cc
	#	cc += 1

	pos_s = np.array(pos_s,dtype="int")
	neg_s = np.array(neg_s,dtype="int")
	pos_nl = np.array(pos_nl,dtype="int")
	neg_nl = np.array(neg_nl,dtype="int")
	pos_ed = np.array(pos_ed,dtype="float")
	neg_ed = np.array(neg_ed,dtype="float")
	pos_cd = np.array(pos_cd,dtype="float")
	neg_cd = np.array(neg_cd,dtype="float")
	
	
	
	print pos_ed
	print "##"
	print neg_ed
	#print "##"
	
	np.savez('data_qpd/'+'pos_s.npz',pos_s)
	np.savez('data_qpd/'+'pos_nl.npz',pos_nl)
	np.savez('data_qpd/'+'pos_ed.npz',pos_ed)
	np.savez('data_qpd/'+'pos_cd.npz',pos_cd)
	
	np.savez('data_qpd/'+'neg_s.npz',neg_s)
	np.savez('data_qpd/'+'neg_nl.npz',neg_nl)
	np.savez('data_qpd/'+'neg_ed.npz',neg_ed)
	np.savez('data_qpd/'+'neg_cd.npz',neg_cd)
	
	#'''
#print "dataset created!"

