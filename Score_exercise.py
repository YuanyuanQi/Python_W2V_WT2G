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

if __name__ == "__main__":
	Filepath = "/home/echo/Documents/word2vec/CNN_keras_demo/query_docs/King Hussein, peace/WT23-B12-229_0.csv"
	with open(Filepath,'rb') as f:
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
				docnolabel_linenum = row[0]+"_"+str(row[1])
				if Dict.has_key(docnolabel_linenum):
					DistList.append(float(row[2]))
					Dict[docnolabel_linenum] = DistList
					continue
				else:
					DistList = []
					DistList.append(float(row[2]))
					Dict[docnolabel_linenum] = DistList
					continue
		items = Dict.items() 
		items.sort()
		csvfile = open("/home/echo/Documents/word2vec/CNN_keras_demo/WT23-B12-229_0_new.csv","wb")
		writer = csv.writer(csvfile)
		writer.writerow(["docnolabel_linenum","King","Hussein","peace","average"])
		for key ,value in items:
			value.append(sum(value)/len(value))
			List =[]
			List.append(key)
			List.append(value)
			writer.writerow(List)
