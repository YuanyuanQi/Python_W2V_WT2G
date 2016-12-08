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
#this code we are using to calculate the score from the query_docs/ path
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
'''
def Score_Idf(FilePath,DocsDict,DocsLen,DocFrqDict,numberOfDocuments,averageDocumentLength):
	Queryfiles = os.listdir(Filepath)
	for Queryfile in Queryfiles:
		queryterms = Queryfile.strip().split(" ")
		query= []
		for term in queryterms:
			query.append(term.strip(","))
			#print query
		if len(query)>1:
			FilePath = os.path.join(Filepath,Queryfile)
			FScore = FilePath.replace("query_docs","Score_docs")
			csvfiles =  os.listdir(FilePath)
			ScoreDict = {}
			Score1Dict = {}
			for csvf in csvfiles:
				if csvf.find("new") != -1 :
					continue
				elif csvf.find("idf") != -1:
					continue
				else:
					docno_label = csvf.replace(".csv","")
					filepath = os.path.join(FilePath,csvf)
					csv_n= csvf.replace(".csv","_idf.csv")
					output =os.path.join(FilePath,csv_n)
					with open(filepath,'rb') as f:
						f_csv = csv.reader(f)
						headers = next(f_csv)
						count = 0
						Dict={}
						DistList =[]
						query_terms= []
						for row in f_csv:
							count = count +1
							if count<1:
								continue
							else:
								if len(row)>2:
									docnolabel_linenum = row[0]+"_"+str(row[1])
									docno = row[0].replace("/","_")
									vocab = DocsDict[docno_label]
									if Dict.has_key(docnolabel_linenum):
										term = row[3]
										if vocab.has_key(term):
											tf = vocab[term]
										else:
											tf = 0
										if DocFrqDict.has_key(term):
											documentFrequency= DocFrqDict[term]
										else:
											documentFrequency = 0
										Idf = math.log((numberOfDocuments - documentFrequency + 0.5) / (documentFrequency + 0.5))
											#print tf,documentFrequency
										DistList.append(float(row[2])*Idf)
											#DistList.append(tf)
											#DistList.append(float(row[2]))
										Dict[docnolabel_linenum] = DistList
										continue
									else:
										term = row[3]
										if vocab.has_key(term):
											tf = vocab[term]
										else:
											tf = 0
										if DocFrqDict.has_key(term):
											documentFrequency= DocFrqDict[term]
										else:
											documentFrequency = 0
										Idf = math.log((numberOfDocuments - documentFrequency + 0.5) / (documentFrequency + 0.5))
											#print tf,documentFrequency
										DistList = []
										DistList.append(float(row[2])*Idf)
											#DistList.append(tf)
											#DistList.append(float(row[2]))
										Dict[docnolabel_linenum] = DistList
						#key is docnolabel_linenum
						#value is list[term1_dist,term2_dist]
					items = Dict.items() 
					items.sort()
					fo_csv = open(output,"wb")
					writer = csv.writer(fo_csv)
					Header = ["docnolabel_linenum"]+query+["Aver_idf"]
					writer.writerow(Header)
					for key ,value in items:
						value.append(sum(value)/len(value))
						List =[]
						List.append(key)
						for item in value:
							List.append(item)
						writer.writerow(List)
					if len(items) >0:
							#items(docnolabel_linum:term1_dist.term2_dist,average)
						score,score1 = Score(items)
							#score =Score_Window(items,D)#
						ScoreDict[docno_label] = score
						Score1Dict[docno_label] = score1
					else:
						print filepath
					fo_csv.close()
				saveCsvFile(FScore+"/ScoreDict_idf.csv",ScoreDict)
				saveCsvFile(FScore+"/Score1Dict_idf.csv",Score1Dict)
		else:
			#sigle query term
			FilePath = os.path.join(Filepath,Queryfile)
			FScore = FilePath.replace("query_docs","Score_docs")
			csvfiles =  os.listdir(FilePath)
			ScoreDict = {}
			Score1Dict = {}
			for csvf in csvfiles:
				docno_label = csvf.replace(".csv","")
				filepath = os.path.join(FilePath,csvf)
					#print filepath
				with open(filepath,'rb') as f:
					f_csv = csv.reader(f)
					headers = next(f_csv)
					count = 0
					List =[]
					for row in f_csv:
						count = count +1
						if count<1:
							continue
						else:
							if len(row)>2:
								term = row[3]
								if vocab.has_key(term):
									tf = vocab[term]
								else:
									tf = 0
								if DocFrqDict.has_key(term):
									documentFrequency= DocFrqDict[term]
								else:
									documentFrequency = 0
								Idf = math.log((numberOfDocuments - documentFrequency + 0.5) / (documentFrequency + 0.5))
											#print tf,documentFrequency
									#DistList.append(float(row[2])*Idf)	
								List.append(float(row[2])*Idf)
									#List.append(float(row[2]))
					aver_doc = sum(List)/len(List)
					#print aver_doc
					List_n =[]
					for item in List:
						if (item*item - aver_doc*aver_doc) > aver_doc:
							List_n.append(item)
					score = sum(List_n)
					score1 = sum(List_n)-len(List_n)*aver_doc
				ScoreDict[docno_label] = score
				Score1Dict[docno_label] = score1
			saveCsvFile(FScore+"/ScoreDict_idf.csv",ScoreDict)
			saveCsvFile(FScore+"/Score1Dict_idf.csv",Score1Dict)


def Score_distD(FilePath,DocsDict,DocsLen,DocFrqDict,numberOfDocuments,averageDocumentLength,D):
	Queryfiles = os.listdir(Filepath)
		for Queryfile in Queryfiles:
			queryterms = Queryfile.strip().split(" ")
			query= []
			for term in queryterms:
				query.append(term.strip(","))
			#print query
			if len(query)>1:
				FilePath = os.path.join(Filepath,Queryfile)
				FScore = FilePath.replace("query_docs","Score_docs")
				csvfiles =  os.listdir(FilePath)
				ScoreDict = {}
				Score1Dict = {}
				for csvf in csvfiles:
					if csvf.find("new") != -1 :
						continue
					elif csvf.find("idf") != -1:
						continue
					else:
						docno_label = csvf.replace(".csv","")
						filepath = os.path.join(FilePath,csvf)
						csv_n= csvf.replace(".csv","_"+str(D)+".csv")
						output =os.path.join(FilePath,csv_n)
						with open(filepath,'rb') as f:
							f_csv = csv.reader(f)
							headers = next(f_csv)
							count = 0
							Dict={}
							DistList =[]
							query_terms= []
							for row in f_csv:
								count = count +1
								if count<1:
									continue
								else:
									if len(row)>2:
										docnolabel_linenum = row[0]+"_"+str(row[1])
										docno = row[0].replace("/","_")
										vocab = DocsDict[docno_label]
										if Dict.has_key(docnolabel_linenum):
											term = row[3]
											if vocab.has_key(term):
												tf = vocab[term]
											else:
												tf = 0
											if DocFrqDict.has_key(term):
												documentFrequency= DocFrqDict[term]
											else:
												documentFrequency = 0
											Idf = math.log((numberOfDocuments - documentFrequency + 0.5) / (documentFrequency + 0.5))
											#print tf,documentFrequency
											#DistList.append(float(row[2])*Idf)
											#DistList.append(tf)
											DistList.append(float(row[2]))
											Dict[docnolabel_linenum] = DistList
											continue
										else:
											term = row[3]
											if vocab.has_key(term):
												tf = vocab[term]
											else:
												tf = 0
											if DocFrqDict.has_key(term):
												documentFrequency= DocFrqDict[term]
											else:
												documentFrequency = 0
											Idf = math.log((numberOfDocuments - documentFrequency + 0.5) / (documentFrequency + 0.5))
											#print tf,documentFrequency
											DistList = []
											#DistList.append(float(row[2])*Idf)
											#DistList.append(tf)
											DistList.append(float(row[2]))
											Dict[docnolabel_linenum] = DistList
						#key is docnolabel_linenum
						#value is list[term1_dist,term2_dist]					continue
						items = Dict.items() 
						items.sort()
						fo_csv = open(output,"wb")
						writer = csv.writer(fo_csv)
						Header = ["docnolabel_linenum"]+query+["Aver_"+str(D)]
						writer.writerow(Header)
						for key ,value in items:
							value.append(sum(value)/len(value))
							List =[]
							List.append(key)
							for item in value:
								List.append(item)
							writer.writerow(List)
						if len(items) >0:
							#items(docnolabel_linum:term1_dist.term2_dist,average)
							score,score1 = Score_distD(items,D)
							ScoreDict[docno_label] = score
							Score1Dict[docno_label] = score1
						else:
							print filepath
						fo_csv.close()
					saveCsvFile(FScore+"/ScoreDict_"+str(D)+".csv",ScoreDict)
					saveCsvFile(FScore+"/Score1Dict_"+str(D)+".csv",Score1Dict)
			else:
			#sigle query term
				FilePath = os.path.join(Filepath,Queryfile)
				FScore = FilePath.replace("query_docs","Score_docs")
				csvfiles =  os.listdir(FilePath)
				ScoreDict = {}
				Score1Dict = {}
				for csvf in csvfiles:
					docno_label = csvf.replace(".csv","")
					filepath = os.path.join(FilePath,csvf)
					#print filepath
					with open(filepath,'rb') as f:
						f_csv = csv.reader(f)
						headers = next(f_csv)
						count = 0
						List =[]
						for row in f_csv:
							count = count +1
							if count<1:
								continue
							else:
								if len(row)>2:
									term = row[3]
									if vocab.has_key(term):
										tf = vocab[term]
									else:
										tf = 0
									if DocFrqDict.has_key(term):
										documentFrequency= DocFrqDict[term]
									else:
										documentFrequency = 0
									Idf = math.log((numberOfDocuments - documentFrequency + 0.5) / (documentFrequency + 0.5))
											#print tf,documentFrequency
									#DistList.append(float(row[2])*Idf)	
									#List.append(float(row[2])*Idf)
									List.append(float(row[2]))
						aver_doc = sum(List)/len(List)
						#print aver_doc
						List_n =[]
						for item in List:
							if (item - D) > aver_doc:
								List_n.append(item)
						score = sum(List_n)
						score1 = sum(List_n)-len(List_n)*aver_doc
					ScoreDict[docno_label] = score
					Score1Dict[docno_label] = score1
				saveCsvFile(FScore+"/ScoreDict_"+str(D)+".csv",ScoreDict)
				saveCsvFile(FScore+"/Score1Dict_"+str(D)+".csv",Score1Dict)


def Score_distnew(FilePath,DocsDict,DocsLen,DocFrqDict,numberOfDocuments,averageDocumentLength):
	Queryfiles = os.listdir(Filepath)
	for Queryfile in Queryfiles:
		queryterms = Queryfile.strip().split(" ")
		query= []
		for term in queryterms:
			query.append(term.strip(","))
		#print query
		if len(query)>1:
			FilePath = os.path.join(Filepath,Queryfile)
				FScore = FilePath.replace("query_docs","Score_docs")
				csvfiles =  os.listdir(FilePath)
				ScoreDict = {}
				Score1Dict = {}
				for csvf in csvfiles:
					if csvf.find("new") != -1 :
						continue
					elif csvf.find("idf") != -1:
						continue
					else:
						docno_label = csvf.replace(".csv","")
						filepath = os.path.join(FilePath,csvf)
						csv_n= csvf.replace(".csv","_new.csv")
						output =os.path.join(FilePath,csv_n)
						with open(filepath,'rb') as f:
							f_csv = csv.reader(f)
							headers = next(f_csv)
							count = 0
							Dict={}
							DistList =[]
							query_terms= []
							for row in f_csv:
								count = count +1
								if count<1:
									continue
								else:
									if len(row)>2:
										docnolabel_linenum = row[0]+"_"+str(row[1])
										docno = row[0].replace("/","_")
										vocab = DocsDict[docno_label]
										if Dict.has_key(docnolabel_linenum):
											term = row[3]
											if vocab.has_key(term):
												tf = vocab[term]
											else:
												tf = 0
											if DocFrqDict.has_key(term):
												documentFrequency= DocFrqDict[term]
											else:
												documentFrequency = 0
											Idf = math.log((numberOfDocuments - documentFrequency + 0.5) / (documentFrequency + 0.5))
											#print tf,documentFrequency
											#DistList.append(float(row[2])*Idf)
											#DistList.append(tf)
											DistList.append(float(row[2]))
											Dict[docnolabel_linenum] = DistList
											continue
										else:
											term = row[3]
											if vocab.has_key(term):
												tf = vocab[term]
											else:
												tf = 0
											if DocFrqDict.has_key(term):
												documentFrequency= DocFrqDict[term]
											else:
												documentFrequency = 0
											Idf = math.log((numberOfDocuments - documentFrequency + 0.5) / (documentFrequency + 0.5))
											#print tf,documentFrequency
											DistList = []
											#DistList.append(float(row[2])*Idf)
											#DistList.append(tf)
											DistList.append(float(row[2]))
											Dict[docnolabel_linenum] = DistList
						#key is docnolabel_linenum
						#value is list[term1_dist,term2_dist]					continue
						items = Dict.items() 
						items.sort()
						fo_csv = open(output,"wb")
						writer = csv.writer(fo_csv)
						Header = ["docnolabel_linenum"]+query+["Aver_new"]
						writer.writerow(Header)
						for key ,value in items:
							value.append(sum(value)/len(value))
							List =[]
							List.append(key)
							for item in value:
								List.append(item)
							writer.writerow(List)
						if len(items) >0:
							#items(docnolabel_linum:term1_dist.term2_dist,average)
							score,score1 = Score_N(items)
							#score =Score_Window(items,D)#
							ScoreDict[docno_label] = score
							Score1Dict[docno_label] = score1
						else:
							print filepath
						fo_csv.close()
					saveCsvFile(FScore+"/ScoreDict_new.csv",ScoreDict)
					saveCsvFile(FScore+"/Score1Dict_new.csv",Score1Dict)
			else:
			#sigle query term
				FilePath = os.path.join(Filepath,Queryfile)
				FScore = FilePath.replace("query_docs","Score_docs")
				csvfiles =  os.listdir(FilePath)
				ScoreDict = {}
				Score1Dict = {}
				for csvf in csvfiles:
					docno_label = csvf.replace(".csv","")
					filepath = os.path.join(FilePath,csvf)
					#print filepath
					with open(filepath,'rb') as f:
						f_csv = csv.reader(f)
						headers = next(f_csv)
						count = 0
						List =[]
						for row in f_csv:
							count = count +1
							if count<1:
								continue
							else:
								if len(row)>2:
									term = row[3]
									if vocab.has_key(term):
										tf = vocab[term]
									else:
										tf = 0
									if DocFrqDict.has_key(term):
										documentFrequency= DocFrqDict[term]
									else:
										documentFrequency = 0
									Idf = math.log((numberOfDocuments - documentFrequency + 0.5) / (documentFrequency + 0.5))
											#print tf,documentFrequency
									#DistList.append(float(row[2])*Idf)	
									#List.append(float(row[2])*Idf)
									List.append(float(row[2]))
						aver_doc = sum(List)/len(List)
						#print aver_doc
						List_n =[]
						for item in List:
							if (item*item - aver_doc*aver_doc) > aver_doc:
								List_n.append(item)
						score = sum(List_n)
						score1 = sum(List_n)-len(List_n)*aver_doc
					ScoreDict[docno_label] = score
					Score1Dict[docno_label] = score1
				saveCsvFile(FScore+"/ScoreDict_new.csv",ScoreDict)
				saveCsvFile(FScore+"/Score1Dict_new.csv",Score1Dict)


def Score_Window(FilePath,DocsDict,DocsLen,DocFrqDict,numberOfDocuments,averageDocumentLength):
	Queryfiles = os.listdir(Filepath)
		for Queryfile in Queryfiles:
			queryterms = Queryfile.strip().split(" ")
			query= []
			for term in queryterms:
				query.append(term.strip(","))
			#print query
			if len(query)>1:
				FilePath = os.path.join(Filepath,Queryfile)
				FScore = FilePath.replace("query_docs","Score_docs")
				csvfiles =  os.listdir(FilePath)
				ScoreDict = {}
				Score1Dict = {}
				for csvf in csvfiles:
					if csvf.find("new") != -1 :
						continue
					elif csvf.find("idf") != -1:
						continue
					else:
						docno_label = csvf.replace(".csv","")
						filepath = os.path.join(FilePath,csvf)
						csv_n= csvf.replace(".csv","_wind.csv")
						output =os.path.join(FilePath,csv_n)
						with open(filepath,'rb') as f:
							f_csv = csv.reader(f)
							headers = next(f_csv)
							count = 0
							Dict={}
							DistList =[]
							query_terms= []
							for row in f_csv:
								count = count +1
								if count<1:
									continue
								else:
									if len(row)>2:
										docnolabel_linenum = row[0]+"_"+str(row[1])
										docno = row[0].replace("/","_")
										vocab = DocsDict[docno_label]
										if Dict.has_key(docnolabel_linenum):
											term = row[3]
											if vocab.has_key(term):
												tf = vocab[term]
											else:
												tf = 0
											if DocFrqDict.has_key(term):
												documentFrequency= DocFrqDict[term]
											else:
												documentFrequency = 0
											Idf = math.log((numberOfDocuments - documentFrequency + 0.5) / (documentFrequency + 0.5))
											#print tf,documentFrequency
											#DistList.append(float(row[2])*Idf)
											#DistList.append(tf)
											DistList.append(float(row[2]))
											Dict[docnolabel_linenum] = DistList
											continue
										else:
											term = row[3]
											if vocab.has_key(term):
												tf = vocab[term]
											else:
												tf = 0
											if DocFrqDict.has_key(term):
												documentFrequency= DocFrqDict[term]
											else:
												documentFrequency = 0
											Idf = math.log((numberOfDocuments - documentFrequency + 0.5) / (documentFrequency + 0.5))
											#print tf,documentFrequency
											DistList = []
											#DistList.append(float(row[2])*Idf)
											#DistList.append(tf)
											DistList.append(float(row[2]))
											Dict[docnolabel_linenum] = DistList
						#key is docnolabel_linenum
						#value is list[term1_dist,term2_dist]
						items = Dict.items() 
						items.sort()
						fo_csv = open(output,"wb")
						writer = csv.writer(fo_csv)
						Header = ["docnolabel_linenum"]+query+["Aver_wind"]
						writer.writerow(Header)
						for key ,value in items:
							value.append(sum(value)/len(value))
							List =[]
							List.append(key)
							for item in value:
								List.append(item)
							writer.writerow(List)
						if len(items) >0:
							#items(docnolabel_linum:term1_dist.term2_dist,average)
							#score,score1 = Score_N(items)
							score =Score_Window(items,D)#
							ScoreDict[docno_label] = score
							#Score1Dict[docno_label] = score1
						else:
							print filepath
						fo_csv.close()
					saveCsvFile(FScore+"/ScoreDict_wind.csv",ScoreDict)
					#saveCsvFile(FScore+"/Score1Dict_new.csv",Score1Dict)
			else:
			#sigle query term
				FilePath = os.path.join(Filepath,Queryfile)
				FScore = FilePath.replace("query_docs","Score_docs")
				csvfiles =  os.listdir(FilePath)
				ScoreDict = {}
				Score1Dict = {}
				for csvf in csvfiles:
					docno_label = csvf.replace(".csv","")
					filepath = os.path.join(FilePath,csvf)
					#print filepath
					with open(filepath,'rb') as f:
						f_csv = csv.reader(f)
						headers = next(f_csv)
						count = 0
						List =[]
						for row in f_csv:
							count = count +1
							if count<1:
								continue
							else:
								if len(row)>2:
									term = row[3]
									if vocab.has_key(term):
										tf = vocab[term]
									else:
										tf = 0
									if DocFrqDict.has_key(term):
										documentFrequency= DocFrqDict[term]
									else:
										documentFrequency = 0
									Idf = math.log((numberOfDocuments - documentFrequency + 0.5) / (documentFrequency + 0.5))
									List.append(float(row[2]))
						aver_doc = sum(List)/len(List)
						local = 
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
					ScoreDict[docno_label] = score
					#Score1Dict[docno_label] = score1
				saveCsvFile(FScore+"/ScoreDict_wind.csv",ScoreDict)#'''


def Score_WindowN(Filepath):
	Queryfiles = os.listdir(Filepath)
	for Queryfile in Queryfiles:
		queryterms = Queryfile.strip().split(" ")
		query= []
		for term in queryterms:
			query.append(term.strip(","))
			#print query
		if len(query)>0:
			FilePath = os.path.join(Filepath,Queryfile)
			FScore = FilePath.replace("query_docs","Score_docs")
			csvfiles =  os.listdir(FilePath)
			ScoreDict = {}
			Score1Dict = {}
			for csvf in csvfiles:
				if csvf.find("new") != -1 :
					continue
				elif csvf.find("idf") != -1:
					continue
				else:
					docno_label = csvf.replace(".csv","")
					filepath = os.path.join(FilePath,csvf)
					csv_n= csvf.replace(".csv","_windnew.csv")
					output =os.path.join(FilePath,csv_n)
					with open(filepath,'rb') as f:
						docno_label = filepath.split("/")[len(filepath.split("/"))-1].replace(".csv","")
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
								if min(item[1])>0.6:
									score = score + sum(item[1])
									Dict1[docno_label+"_"+str(item[0])] = item[1]
						ScoreDict[docno_label] = score
					saveCsvFile(FScore+"/ScoreDict_windNew06.csv",ScoreDict)#'''



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
	Score_WindowN(Filepath)