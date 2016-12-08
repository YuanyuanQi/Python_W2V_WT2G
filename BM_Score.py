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
# this code is to generate BM25 score from the part query pos/neg files 
#

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

'''
double k_1 = 1.2d;
		double k_3 = 8d;
		double b_1 = mutralParameters.beta;
		double K = k_1 * ((1 - b_1) + b_1 * docLength / averageDocumentLength) + tf;
		double BM25 = (tf * (k_3 + 1d) * keyFrequency / ((k_3 + keyFrequency) * K))
	            * i.log((numberOfDocuments - documentFrequency + 0.5d) / (documentFrequency + 0.5d));
'''
#docLength is length of documents without repeated words
#numberOfDocuments is number of documents in corpus
#documentFrequency is number of docments which query terms appeared
#keyFrequency is the frequency of term that appeated in query 

def CleanSentence(DictPath,SList,clean_string=True):
	DocDict = pickle.load(open(DictPath,'rb'))
	st = DictPath.split("/")[len(DictPath.split("/"))-1]
	docno = st.split("_")[1]+"_"+st.split("_")[2]
	vocab = defaultdict(float)
	Dict ={}
	for key,value in DocDict.items():
		rev = []
		rev.append(value.strip())
		if clean_string:
			orig_rev = clean_str(" ".join(rev))
		else:
			orig_rev = " ".join(rev).lower()
		orig_rev_list= orig_rev.split()
		new_list = [w for w in orig_rev_list if not w in SList]#remove the stopwords in the sentences
		orig_rev_new =" ".join(new_list)
		#words = set(orig_rev_new.split())#remove repeated items in list
		words = orig_rev_new.split()#do not remove the repeated items in the list
		for word in words:
			if word in SList:
				print word + "  is in the vocab"
			vocab[word] += 1
		
	Dict[docno]= vocab
	return Dict
	

#get a big dict for whole files in /query_docs which key is docno,value is the words dict(key is word,value is tf of the key)
def GetParameters(SList):
	Filepath = "/home/echo/Documents/word2vec/CNN_keras_demo/query_docs/"
	Queryfiles = os.listdir(Filepath)
	DocsDict ={}
	for Queryfile in Queryfiles:
		queryterms = Queryfile.strip().split(" ")
		query= []
		for term in queryterms:
			query.append(term.strip(","))
		FilePath = os.path.join(Filepath,Queryfile)
		csvfiles =  os.listdir(FilePath)
		for csvf in csvfiles:
			if csvf.find("new") != -1:
				continue
			else:
				docno_label = csvf.replace(".csv","")
				Dfilename = Queryfile+"_"+docno_label
				#print Dfilename
				label = docno_label.split("_")[1]
				if label =="1":
					DictPath = "/home/echo/Documents/WT2G/Sentence/POS_new/"+Dfilename
					DocsDict.update(CleanSentence(DictPath,SList,clean_string=True))	
				else:
					DictPath = "/home/echo/Documents/WT2G/Sentence/NEG_new/"+Dfilename
					DocsDict.update(CleanSentence(DictPath,SList,clean_string=True))		
	return DocsDict
	
def GetDocsNumCorpus(SList,clean_string=True):
	DocsNumCorpus =0
	DocsLen ={}
	DirPath = "/home/echo/Documents/WT2G/"
	Files = os.listdir(DirPath)
	for File in Files:
		#print File
		if File.find("WT") != -1:
			SecPath = os.path.join(DirPath,File)
			#print SecPath
			if os.path.exists(SecPath):
				DocsDictPath =os.path.join(SecPath,"DOCs_TEXT_Dict")
				#print DocsDictPath
				DocsDict = pickle.load(open(DocsDictPath,"rb"))#Dict_docno_text[docno]= text
				#DocsNumCorpus = DocsNumCorpus+len(DocsDict)
				for key,value in DocsDict.items():
					revdoc = []
					for line,sent in value.items():
						rev = []
						rev.append(sent.strip())
						if clean_string:
							orig_rev = clean_str(" ".join(rev))
						else:
							orig_rev = " ".join(rev).lower()
						revdoc.append(orig_rev.split())
					orig_rev_list=[]
					for item in revdoc:
						for word in item:
							orig_rev_list.append(word)
					#print orig_rev_list
					new_list = [w for w in orig_rev_list if not w in SList]#remove the stopwords in the sentences
					orig_rev_new =" ".join(new_list)
					words = set(orig_rev_new.split())#remove repeated items in list
					DocsLen[key]=len(words)
	print len(DocsLen)
	return DocsLen

def BM_Score(docLength,averageDocumentLength,numberOfDocuments,documentFrequency,tf):
	k_1 = 1.2
	k_3 = 8.0
	b_1 = 0.35
	keyFrequency=1.0
	K = k_1 * ((1 - b_1) + b_1 * docLength / averageDocumentLength) + tf
	Idf = math.log((numberOfDocuments - documentFrequency + 0.5) / (documentFrequency + 0.5))
	BM25 = (tf * (k_3 + 1.0) * keyFrequency / ((k_3 + keyFrequency) * K))* Idf
	return BM25

def scoring(DocsDict,numberOfDocuments,averageDocumentLength,DocFrqDict,DocsLen):
#def scoring():
	Filepath = "/home/echo/Documents/word2vec/CNN_keras_demo/query_docs/"
	Queryfiles = os.listdir(Filepath)
	#DocsDict ={}
	for Queryfile in Queryfiles:
		BM25 ={}
		queryterms = Queryfile.strip().split(" ")
		query= []
		for term in queryterms:
			query.append(term.strip(","))
		FilePath = os.path.join(Filepath,Queryfile)
		csvfiles =  os.listdir(FilePath)
		for csvf in csvfiles:
			if csvf.find("new") != -1:
				continue
			else:
				docno_label = csvf.replace(".csv","")#docno_label = WT13-B23-171_1
				docno = docno_label.split("_")[0]
				docLength = DocsLen[docno]
				if DocsDict.has_key(docno_label):
					vocab = DocsDict[docno_label]
					BM25[docno_label]= SumQryTrms(query,vocab,numberOfDocuments,averageDocumentLength,DocFrqDict,docLength)
				#csv_n= csvf.replace(".csv","_bm25.csv")
		output =os.path.join("/home/echo/Documents/word2vec/CNN_keras_demo/Score_docs/"+Queryfile,"Docnolabel_bm25.csv")
		fo_csv = open(output,"wb")
		writer = csv.writer(fo_csv)
		Header = ["docno_label","bm25"]
		writer.writerow(Header)
		items = sorted(BM25.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)#rank by value from lagest to lowest
		for key ,value in items:
			List =[]
			List.append(key)
			List.append(value)
			writer.writerow(List)
		fo_csv.close()
				
				
def SumQryTrms(querylist,vocab,numberOfDocuments,averageDocumentLength,DocFrqDict,docLength):
	score =0
	for term in querylist:
		if vocab.has_key(term):
			tf = vocab[term]
			if DocFrqDict.has_key(term):
				documentFrequency= DocFrqDict[term]
				score= score+ BM_Score(docLength,averageDocumentLength,numberOfDocuments,documentFrequency,tf)
			else:
				print "NOT find query term : "+term+" in DoFrqDict"
		else:
			print "NOT find query term : "+term+" in Vocab"
			
	return score
			
			
			
if __name__ == "__main__":
	stopwords_file = "/home/echo/Documents/word2vec/CNN_keras_demo/stopword-list.txt"
	SList = get_S(stopwords_file)
	if os.path.exists("/home/echo/Documents/word2vec/CNN_keras_demo/DocsDict"):
		print "Loading DocsDict"
		DocsDict= pickle.load(open("/home/echo/Documents/word2vec/CNN_keras_demo/DocsDict","rb"))
	else:
		print "Generating DocsDict"
		DocsDict= GetParameters(SList)#key is target docnos(WT02-B33-20_1) not whole docno of corpus ,value is dict (key is word,value is tf og word)
		saveObjToFile("DocsDict",DocsDict)
	if os.path.exists("/home/echo/Documents/word2vec/CNN_keras_demo/DocsLen"):
		print "Loading DocsLen"
		DocsLen= pickle.load(open("/home/echo/Documents/word2vec/CNN_keras_demo/DocsLen","rb"))
	else:
		print "Generating DocsLen"
		DocsLen=GetDocsNumCorpus(SList,clean_string=True)#key is docno:"WT01-B11-196",value is the length of docno without stopwords and repeated words but not stemmed
		saveObjToFile("DocsLen",DocsLen)
	numberOfDocuments = len(DocsLen)
	averageDocumentLength = sum(DocsLen.values())/len(DocsLen)
	DocFrqDict = pickle.load(open("/home/echo/Documents/WT2G/Sentence/WordTDF_Dict","rb"))#key is word,value is number of docments which words appeared
	print "Scoring!"
	scoring(DocsDict,numberOfDocuments,averageDocumentLength,DocFrqDict,DocsLen)
	print "Done"
	#'''
	
	
	
	
	
	
