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
#this code we use to generate the score of csvfile into the query_docs file by the dict
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

if __name__ == "__main__":
	GaoDict1 = pickle.load(open("/home/echo/Documents/word2vec/CNN_keras_demo/GaoDict1","rb"))
	GaoDict2 = pickle.load(open("/home/echo/Documents/word2vec/CNN_keras_demo/GaoDict2","rb"))
	GaoDict3 = pickle.load(open("/home/echo/Documents/word2vec/CNN_keras_demo/GaoDict3","rb"))
	#GaoDict1[docname.split("_")[1]+"/"+docname.split("_")[2]+"_"+str(rev["line_num"])] = [key+"_"+word,max(value)]
	query_termlist,dnlist,qtlist,lnlist,lblist = [],[],[],[],[]
	dlDict={}
	for key ,value in GaoDict1.items():
		docno = key.split("_")[0].split("/")[0]
		label = key.split("_")[0].split("/")[1]
		query_termlist.append(value[0].split("_")[0])
		line_num = int(key.split("_")[1])
		if docno in dnlist:
			if label in lblist:
				if line_num not in lnlist:
					lnlist.append(line_num)
				dlDict [docno+"/"+label]= sorted(lnlist)
			else:
				lblist.append(label)
				if line_num not in lnlist:
					lnlist.append(line_num)
				dlDict [docno+"/"+label]= sorted(lnlist)
		else:
			dnlist.append(docno)
			if label in lblist:
				if line_num not in lnlist:
					lnlist.append(line_num)
				dlDict [docno+"/"+label]= sorted(lnlist)
			else:
				lblist.append(label)
				if line_num not in lnlist:
					lnlist.append(line_num)
				dlDict [docno+"/"+label]= sorted(lnlist)#'''
	#print list(set(query_termlist))
	#dnList=list(set(dnlist))
	print "Writing CSV Files"
	for docnolabel,linenumlist in dlDict.items():
		csvfile = file('/home/echo/Documents/word2vec/CNN_keras_demo/King Hussein, peace/'+docnolabel.replace('/','_')+'.csv', 'wb')
		#.replace('/','_')
		writer = csv.writer(csvfile)
		writer.writerow(["docno/label","line_num","dist","query_term","word"])
		for linenum in linenumlist:
			rowlist=[]
			key = docnolabel+"_"+str(linenum)
			if GaoDict1.has_key(key):
				rowlist.append(docnolabel)
				rowlist.append(linenum)
				rowlist.append(GaoDict1[key][1])
				rowlist.append(GaoDict1[key][0].split("_")[0])
				rowlist.append(GaoDict1[key][0].split("_")[1])
				#print "**************************************************"
				#print rowlist
				writer.writerow(rowlist)
				rowlist=[]
			if GaoDict2.has_key(key):
				#print GaoDict1[key][0].split("_")[0]
				rowlist.append(docnolabel)
				rowlist.append(linenum)
				rowlist.append(GaoDict2[key][1])
				rowlist.append(GaoDict2[key][0].split("_")[0])
				rowlist.append(GaoDict2[key][0].split("_")[1])
				#print "**************************************************"
				#print rowlist
				writer.writerow(rowlist)
				rowlist=[]
			if GaoDict3.has_key(key):
				#print GaoDict1[key][0].split("_")[0]
				rowlist.append(docnolabel)
				rowlist.append(linenum)
				rowlist.append(GaoDict3[key][1])
				rowlist.append(GaoDict3[key][0].split("_")[0])
				rowlist.append(GaoDict3[key][0].split("_")[1])
				#print "**************************************************"
				#print rowlist
				writer.writerow(rowlist)
				rowlist=[]
			else:
				continue
		csvfile.close()
		
		
		
		