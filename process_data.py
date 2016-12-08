import numpy as np
import cPickle
from collections import defaultdict
import sys, re, os
import pandas as pd
import random

def build_data_cv(data_folder, cv=4800, clean_string=True):
	revs = []
	pos_file = data_folder[0]
	neg_file = data_folder[1]
	vocab = defaultdict(float)

	t2t = 0
	symbol = 0
	t2t_neg = 0
	symbol_neg = 0

	with open(pos_file, "rb") as f:
		for line in f:
			t2t += 1       
			rev = []
			rev.append(line.strip())
			if clean_string:
				orig_rev = clean_str(" ".join(rev))
			else:
				orig_rev = " ".join(rev).lower()
			#print "orig_rev"
			#print orig_rev
			words = set(orig_rev.split())
			#print "words"
			#print words
			for word in words:
				vocab[word] += 1
			#print "vocab"
			#print vocab
			if t2t > 4800:
				symbol = 1
			else:
				symbol = 0
			datum  = {"y":1, 
								"text": orig_rev,                             
								"num_words": len(orig_rev.split()),
								"split": symbol}
			revs.append(datum)
	with open(neg_file, "rb") as f:
		for line in f:
			t2t_neg += 1       
			rev = []
			rev.append(line.strip())
			if clean_string:
				orig_rev = clean_str(" ".join(rev))
			else:
				orig_rev = " ".join(rev).lower()
			words = set(orig_rev.split())
			for word in words:
				vocab[word] += 1
			if t2t_neg > 4800:
				symbol_neg = 1
			else:
				symbol_neg = 0
			datum  = {"y":0, 
								"text": orig_rev,                             
								"num_words": len(orig_rev.split()),
								"split": symbol_neg}
			revs.append(datum)
	'''
	train, test, train_y, test_y = [], [], [], []
	for rev in revs:
		sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
		#sent.append(rev["y"])
		if int(rev["split"])==1:
			test.append(sent)
			test_y.append(rev["y"])
		else:
			train.append(sent)
			train_y.append(rev["y"])
		train = np.array(train,dtype="int")
		test = np.array(test,dtype="int")
		train_y = np.array(train,dtype="int")
		test_y = np.array(test,dtype="int")
		np.savez('data_processed/'+'train_x.npz',train)
		np.savez('data_processed/'+'train_y.npz',train_y)
		np.savez('data_processed/'+'test_x.npz',test)
		np.savez('data_processed/'+'test_y.npz',test_y)
	'''	
	return revs, vocab
#sent is  rev["text"] x is list for vector for each word in sent
def get_idx_from_sent(sent, word_idx_map, max_l=56, k=300, filter_h=5):
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
	#print "output is list to instore words which  matched with vector? "
	#print x
	return x
			
    
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

def load_bin_vec(fname, vocab):
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

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
	for word in vocab:
		if word not in word_vecs and vocab[word] >= min_df:
			word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

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







if __name__=="__main__": 
	#print sys.argv   
	#w2v_file = sys.argv[1]  
	w2v_file = "/home/echo/Documents/word2vec/CNN_keras_demo/GoogleNews-vectors-negative300.bin"  
	data_folder = ["rt-polarity.pos","rt-polarity.neg"]    
	print "loading data...",        
	revs, vocab = build_data_cv(data_folder, cv=4800, clean_string=True)
	#print "vocab"
	#print vocab
	#print len(vocab) ==18765
	#revs is list which element is map (each key is diging each line from pos and neg text information)
	#vocab is list which instore all words from neg and pos files
	max_l = np.max(pd.DataFrame(revs)["num_words"])
	#what is pd(panda?)
	print "data loaded!"
	print "number of sentences: " + str(len(revs))
	print "vocab size: " + str(len(vocab))
	print "max sentence length: " + str(max_l)
	print "loading word2vec vectors...",
	w2v = load_bin_vec(w2v_file, vocab)
	print "word2vec loaded!"
	print "num words already in word2vec: " + str(len(w2v))
	add_unknown_words(w2v, vocab)
	W, word_idx_map = get_W(w2v)
	#print " word_idx_map   18765 words"
	#print word_idx_map
	#print type(W)
	#print W.shape
	np.savez('data_qpd/embedding_weights.npz',embedding_weights=W)
	#what the npz file use for?
	f_idx_map = open('data_qpd/w2v_index','w')
	for item in word_idx_map:
		f_idx_map.write(item+'\t'+str(word_idx_map[item])+'\n')
	
	train, test, train_y, test_y = [], [], [], []
	
	random.shuffle(revs)
	for rev in revs:
		sent = get_idx_from_sent(rev["text"], word_idx_map, max_l=56, k=300, filter_h=5)
		#words in line ==rev["text"]
#		beautifully crafted , engaging film making that should attract upscale audiences hungry for quality and a nostalgic , twisty yarn that will keep them guessing
#		output is list to instore words which  matched with index location from index_map
#		[0, 0, 0, 0, 3132, 4244, 9596, 12054, 5943, 6347, 4119, 14263, 808, 18396, 15804, 970, 8554, 16503, 7252, 17918, 9596, 1966, 13317, 6347, 18380,8523, 5012, 3481, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

		#sent.append(rev["y"])
		if int(rev["split"])==1:
			test.append(sent)
			test_y.append(rev["y"])
		else:
			train.append(sent)
			train_y.append(rev["y"])
	#y is lable for train or test data
	#print type(train)
	#print "\n"
	train = np.asarray(train)
	test = np.asarray(test)
	train_y = np.asarray(train_y)
	test_y = np.asarray(test_y)

	#print train_y.shape
	#print test_y.shape
	print "###"
	#cc = 1
	#for item in train:
	#	if len(item) != 59:
	#		print len(item)
	#		print cc
	#	cc += 1

	train = np.array(train,dtype="int")
	test = np.array(test,dtype="int")
	train_y = np.array(train_y,dtype="int")
	test_y = np.array(test_y,dtype="int")
	
	print train_y
	print "##"
	print test_y

	np.savez('data_qpd/'+'train_x.npz',train)
	np.savez('data_qpd/'+'train_y.npz',train_y)
	np.savez('data_qpd/'+'test_x.npz',test)
	np.savez('data_qpd/'+'test_y.npz',test_y)

	#rand_vecs = {}
	#add_unknown_words(rand_vecs, vocab)
	#W2, _ = get_W(rand_vecs)
	#cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
	print "dataset created!"
    
