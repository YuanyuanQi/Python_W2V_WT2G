from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D, Masking
from keras.layers import merge, Convolution1D, MaxPooling1D, Input,Flatten,LSTM
from keras.models import Model, model_from_json, model_from_config
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
import numpy as np
from theano import tensor as T
import theano
from keras.utils import np_utils
from  keras.callbacks import ModelCheckpoint,Callback

def build_model(sent_size,vec_size,embedding_weights,nb_class):
	mask_value = 0
	input_sentence = Input(shape=(sent_size,), dtype='int32')
	embedding = keras.layers.embeddings.Embedding(input_dim = embedding_weights.shape[0],output_dim = vec_size,weights=[embedding_weights])
	#print "input_dim   output_dim    weights    input_sentence=Input(shape=(sent_size,), dtype='int32')  "
	#print embedding_weights.shape[0]
	#print vec_size
	#print embedding_weights
	#print input_sentence
	x = embedding(input_sentence)
	#print "embedding"
	#print x
	x3 = Convolution1D(100, 3,input_shape=(sent_size,vec_size),activation='relu')(x)
	#print "Convolution1D"
	#print x3
	x3 = MaxPooling1D((sent_size-2))(x3)
	#print "MaxPooling1D"
	#print x3
	x3 = Flatten()(x3)
	#print "Flatten"
	#print x3
	x4 = Convolution1D(100, 4,input_shape=(sent_size,vec_size),activation='relu')(x)
	x4 = MaxPooling1D((sent_size-3))(x4)
	x4 = Flatten()(x4)
	x5 = Convolution1D(100, 5,input_shape=(sent_size,vec_size),activation='relu')(x)
	x5 = MaxPooling1D((sent_size-4))(x5)
	x5 = Flatten()(x5)
	out = merge([x3,x4,x5],mode='concat')
	out = Dense(100,activation='tanh',input_dim = 300)(out)
	out = Dropout(0.5)(out)
	#out = Dense(nb_class,activation='softmax')(out)
	out = Dense(1, activation="sigmoid")(out)
	model = Model(input_sentence, out)

	return model


def train_model(x1_train,y_train,embedding_weights,save_to,split,nb_class):
	pair_wise_cnn = build_model(64,300,embedding_weights,nb_class)
	#print "pair_wise_cnn"
	#print pair_wise_cnn
	json_string = pair_wise_cnn.to_json()
	#print "pair_wise_cnn.to_json()"
	#print json_string
	pair_wise_cnn.compile(loss='binary_crossentropy',optimizer='adadelta',metrics=['accuracy'])
	#early_stopping = EarlyStopping(monitor='val_loss', patience=100)
	checkpointer =ModelCheckpoint(filepath="qpd_mouth.hdf5", verbose=1, save_best_only=True, monitor='val_acc',save_weights_only=True)
	pair_wise_cnn.fit([x1_train], [y_train], nb_epoch=1,batch_size=50, shuffle=True, validation_split=split, callbacks=[checkpointer])
	open('qpd_model_architecture.json','w').write(json_string)
	
	return json_string
	

def test_model(x1_test,y_test,embedding_weights,json_string):
	pair_wise_cnn = model_from_json(json_string)
	pair_wise_cnn.load_weights('qpd_mouth.hdf5')
	#print type(pair_wise_cnn.get_weights())
	#for xx in pair_wise_cnn.get_weights():
	#	print xx.shape
	
	pair_wise_cnn.compile(loss='binary_crossentropy',optimizer='adadelta', metrics=['accuracy'])
	score = pair_wise_cnn.evaluate(x1_test, y_test, batch_size=32)
	return score


if __name__ == '__main__':
	save_to = 'model/test_6_20.npz'
	nb_class = 2
	split = 0.1 # stan 2
	embedding_weights = np.load('data_qpd/embedding_weights.npz','rb')['embedding_weights']
	x_train = np.load('data_qpd/train_x.npz','rb')['arr_0']
	print x_train
	y_train = np.load('data_qpd/train_y.npz')['arr_0']
	x_test = np.load('data_qpd/test_x.npz','rb')['arr_0']
	y_test = np.load('data_qpd/test_y.npz')['arr_0']
	print "shape"
	print x_train.shape
	print y_train.shape
	print x_test.shape
	print y_test.shape
	json_string = train_model(x_train,y_train,embedding_weights,save_to,split,nb_class)
	#what is the json_string?
	score = test_model(x_test,y_test,embedding_weights,json_string)
	print score
