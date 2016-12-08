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

#same weight of each term in query
def Score(items):
	List =[]
	List_n = []
	for key ,value in items:
		List.append(float(value[len(value)-1]))
	#List item is the average dist among terms of query
	aver_doc = sum(List)/len(List)
	for item in List:
		if item-aver_doc > 0.0:
			List_n.append(item)
	score = sum(List_n)
	score1 = sum(List_n)-len(List_n)*aver_doc
	return score,score1

def Score_D(items):
	List =[]
	List_n = []
	for key ,value in items:
		List.append(float(value[len(value)-1]))
	#List item is the average dist among terms of query
	aver_doc = sum(List)/len(List)
	for item in List:
		if (item - D ) > aver_doc:
			List_n.append(item)
	score = sum(List_n)
	score1 = sum(List_n)-len(List_n)*aver_doc
	return score,score1

def Score_N(items):
	List =[]
	List_n = []
	for key ,value in items:
		List.append(float(value[len(value)-1]))
	#List item is the average dist among terms of query
	aver_doc = sum(List)/len(List)
	for item in List:
		if (item*item - aver_doc*aver_doc) > aver_doc:
			List_n.append(item)
	score = sum(List_n)
	score1 = sum(List_n)-len(List_n)*aver_doc
	return score,score1

def Score_Window(items):
	List =[]
	List_n = []
	N= 3
	for key ,value in items:
		List.append(float(value[len(value)-1]))
	aver_doc = sum(List)/len(List)
	#print aver_doc
	local = 1
	score = 0.0
	score1 = 0.0
	Dict ={}
	temp = []
	while(local<len(List)):
		temp = List[local-1:local+2]
		if min(temp) >aver_doc:
			Dict[str(local)] = temp
		local = local + 1
		continue
	for key,value in Dict.items():
		score = score +sum(value)
	return score#'''

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

def Score_WindowN(Filepath):
	AllDocs = 0
	Queryfiles = os.listdir(Filepath)
	
	for Queryfile in Queryfiles:
		queryterms = Queryfile.strip().split(" ")
		query= []
		for term in queryterms:
			query.append(term.strip(","))
		#Docs = 0
		if len(query)>0:
			FilePath = os.path.join(Filepath,Queryfile)
			FScore = FilePath.replace("query_docs","Score_docs")
			csvfiles =  os.listdir(FilePath)
			ScoreDict = {}
			Score1Dict = {}
			#print len(csvfiles)/3
			AllDocs =AllDocs + len(csvfiles)/3#the documents number that retrievaled by each query
			for csvf in csvfiles:
				if csvf.find("new") != -1 :
					continue
				elif csvf.find("idf") != -1:
					continue
				else:
					#count = count + 1
					docno_label = csvf.replace(".csv","")
					filepath = os.path.join(FilePath,csvf)
					#csv_n= csvf.replace(".csv","_windnew.csv")
					#output =os.path.join(FilePath,csv_n)
					with open(filepath,'rb') as f:
						docno_label = filepath.split("/")[len(filepath.split("/"))-1].replace(".csv","")
						f_csv = csv.reader(f)
						headers = next(f_csv)
						count = 0
						Dict={}
						ldict ={}
						DistList =[]
						#query_terms= []
						for row in f_csv:
							count = count +1
							if count<1:
								continue
							else:
								if len(row)>2:
									docnolabel_linenum = row[0]+"_"+str(row[1])
									if Dict.has_key(docnolabel_linenum):
										DistList.append(float(row[2]))
										Dict[docnolabel_linenum] = DistList
										ldict[int(row[1])] = DistList
										continue
									else:
										DistList = []
										DistList.append(float(row[2]))
										Dict[docnolabel_linenum] = DistList
										ldict[int(row[1])]= DistList
					#key is docnolabel_linenum
					#value is list[term1_dist,term2_dist....]
						items = ldict.items() 
						items.sort()
						Dlist = []
						for key,value in items:
							for item in value:
								Dlist.append(item)
						Mean = sum(Dlist)/len(Dlist)
						local = 1
						temp =[]
						Dict1 ={}
						score = 0.0
						while(local<len(items)):
							temp = items[local-1:local+2]#window size is 3
							local = local + 1
							for item in temp:
								if min(item[1])>Mean:
									score = score + sum(item[1])
									Dict1[docno_label+"_"+str(item[0])] = item[1]
						ScoreDict[docno_label] = score
					saveCsvFile(FScore+"/ScoreDict_windNew_Mean.csv",ScoreDict)#'''
					#Evaluation(ScoreDict,query)
			#AllDocs = AllDocs + Docs
	#return AllDocs
	
	
	
def evaluationFiles():
	fo_score = open("P@N_Ranking_Result.csv","wb")
	writer = csv.writer(fo_score)
	Header = ["query","Method","P@3","P@5","P@10","P@20","P@50","P@100"]
	writer.writerow(Header)
	List =[]
	Results = "/home/echo/Documents/word2vec/CNN_keras_demo/Score_docs"
	queryfiles = os.listdir(Results)
	for queryfile in queryfiles:
		FilePath = os.path.join(Results,queryfile)
		scorefiles = os.listdir(FilePath)
		for csvf in scorefiles:
			filepath = os.path.join(FilePath,csvf)
			if csvf.find("Docnolabel_bm25") != -1:
				List.append(queryfile)
				with open(filepath,'rb') as f:
					Dict = {}
					f_csv = csv.reader(f)
					headers = next(f_csv)
					for row in f_csv:
						Dict[row[0]] = float(row[1])
					List.append( "BM25")
					items = sorted(Dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
					l=(Evaluation(items,queryfile))
					for item in l:
						List.append(item)
					writer.writerow(List)
					List= []
			elif csvf.find("ScoreDict_wind3") != -1:
				List.append(queryfile)
				with open(filepath,'rb') as f:
					Dict = {}
					f_csv = csv.reader(f)
					headers = next(f_csv)
					for row in f_csv:
						Dict[row[0]] = float(row[1])
					items = sorted(Dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
					List.append( "WID0.3")
					l=(Evaluation(items,queryfile))
					for item in l:
						List.append(item)
					writer.writerow(List)
					List =[]
			elif csvf.find("ScoreDict_windNew0.4") != -1:
				List.append(queryfile)
				with open(filepath,'rb') as f:
					Dict = {}
					f_csv = csv.reader(f)
					headers = next(f_csv)
					for row in f_csv:
						Dict[row[0]] = float(row[1])
					items = sorted(Dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
					List.append( "WID0.4")
					l=(Evaluation(items,queryfile))
					for item in l:
						List.append(item)
					writer.writerow(List)
					List =[]
			elif csvf.find("ScoreDict_windNew05") != -1:
				List.append(queryfile)
				with open(filepath,'rb') as f:
					#print filepath
					Dict = {}
					f_csv = csv.reader(f)
					headers = next(f_csv)
					for row in f_csv:
						Dict[row[0]] = float(row[1])
					items = sorted(Dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
					List.append( "WID0.5")
					l=(Evaluation(items,queryfile))
					for item in l:
						List.append(item)
					writer.writerow(List)
					List =[]
			elif csvf.find("ScoreDict_windNew06") != -1:
				List.append(queryfile)
				with open(filepath,'rb') as f:
					#print filepath
					Dict = {}
					f_csv = csv.reader(f)
					headers = next(f_csv)
					for row in f_csv:
						Dict[row[0]] = float(row[1])
					items = sorted(Dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
					List.append( "WID0.6")
					l=(Evaluation(items,queryfile))
					for item in l:
						List.append(item)
					writer.writerow(List)
					List =[]
			elif csvf.find("ScoreDict_windNew032") != -1:
				List.append(queryfile)
				with open(filepath,'rb') as f:
					Dict = {}
					f_csv = csv.reader(f)
					headers = next(f_csv)
					for row in f_csv:
						Dict[row[0]] = float(row[1])
					items = sorted(Dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
					List.append( "WID0.32")
					l=(Evaluation(items,queryfile))
					for item in l:
						List.append(item)
					writer.writerow(List)
					List =[]
			elif csvf.find("ScoreDict_windNew_Mean") != -1:
				List.append(queryfile)
				with open(filepath,'rb') as f:
					Dict = {}
					f_csv = csv.reader(f)
					headers = next(f_csv)
					for row in f_csv:
						Dict[row[0]] = float(row[1])
					items = sorted(Dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
					List.append( "WID_Mean")
					l=(Evaluation(items,queryfile))
					for item in l:
						List.append(item)
					writer.writerow(List)
					List =[]
	fo_score.close()
		#elif scorefile.find("") != -1:
def Evaluation(ScoreDict,query):
	rgt3 = 0.0
	rgt5 = 0.0
	rgt10 = 0.0
	rgt20 = 0.0
	rgt50 =0.0
	rgt100 = 0.0
	List = []
	for docno_label,score in ScoreDict[:2]:
	 	label = docno_label.split("_")[1]
	 	if label =="1":
	 		rgt3 = rgt3 + 1
	for docno_label,score in ScoreDict[:4]:
	 	label = docno_label.split("_")[1]
	 	if label =="1":
	 		rgt5 = rgt5 + 1
	for docno_label,score in ScoreDict[:9]:
	 	label = docno_label.split("_")[1]
	 	if label =="1":
	 		rgt10 = rgt10 + 1
	for docno_label,score in ScoreDict[:19]:
	 	label = docno_label.split("_")[1]
	 	if label =="1":
	 		rgt20 = rgt20 + 1
	for docno_label,score in ScoreDict[:49]:
	 	label = docno_label.split("_")[1]
	 	if label =="1":
	 		rgt50 = rgt50 + 1
	for docno_label,score in ScoreDict[:99]:
	 	label = docno_label.split("_")[1]
	 	if label =="1":
	 		rgt100 = rgt100 + 1
	P3 = rgt3/3.0
	List.append(P3)
	P5 = rgt5/5.0
	List.append(P5)
	P10 = rgt10/10.0
	List.append(P10)
	P20 = rgt20/20.0
	List.append(P20)
	P50 = rgt50/50.0
	List.append(P50)
	P100 = rgt100/100.0
	List.append(P100)
	#List= [query,method,str(P@3),str(P@5),str(P@10),str(P@20)]
	print "".join(query) + " P@3 : " + str(P3)+ " P@5 : " + str(P5) + " P@10 : " +str(P10)+ " P@20 : " + str(P20)+ " P@50 : " + str(P50)+ " P@100 : " + str(P100)
	return List
	
	
	
def exer():
	#filepath = "/home/echo/Documents/word2vec/CNN_keras_demo/query_docs/salvaging, shipwreck, treasure/WT13-B31-268_0.csv"
	filepath = "/home/echo/Documents/word2vec/CNN_keras_demo/query_docs/salvaging, shipwreck, treasure/WT15-B40-217_1.csv"
	with open(filepath,'rb') as f:
		docno_label = filepath.split("/")[len(filepath.split("/"))-1].replace(".csv","")
		print docno_label
		f_csv = csv.reader(f)
		headers = next(f_csv)
		count = 0
		Dict={}
		ldict ={}
		DistList =[]
		query_terms= []
		for row in f_csv:
			count = count +1
			if count<1:
				continue
			else:
				if len(row)>2:
					docnolabel_linenum = row[0]+"_"+str(row[1])
					if Dict.has_key(docnolabel_linenum):
						DistList.append(float(row[2]))
						Dict[docnolabel_linenum] = DistList
						ldict[int(row[1])] = DistList
						continue
					else:
						DistList = []
						DistList.append(float(row[2]))
						Dict[docnolabel_linenum] = DistList
						ldict[int(row[1])]= DistList
	#key is docnolabel_linenum
	#value is list[term1_dist,term2_dist....]
		items = ldict.items() 
		items.sort()
		local = 1
		temp =[]
		Dict1 ={}
		score = 0.0
		while(local<len(items)):
			temp = items[local-1:local+2]#window size is 3
			local = local + 1
			for item in temp:
				if min(item[1])>0.3:
					score = score + sum(item[1])
					Dict1[docno_label+"_"+str(item[0])] = item[1]
		for key,value in Dict1.items():
			print key + " : ",value
		print score



if __name__ == "__main__":
	FilePath = "/home/echo/Documents/word2vec/CNN_keras_demo/query_docs/"
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
	Score_WindowN(FilePath)
	evaluationFiles()
	
