import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras_tqdm import TQDMNotebookCallback,TQDMCallback
from keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from keras.layers import Dense, Dropout, GaussianNoise, Conv1D,Conv2D, MaxPooling2D, UpSampling2D,Lambda,Layer
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import optimizers
from mainUtil import PlotDataAE,PlotHistory,LoadMnist


class PCA():
    def __init__(self,inputDimentions=784,encoded_dim=3,summary=False,metric='binary_crossentropy'):
        self.summary  = summary
        self.inputDimentions=inputDimentions
        self.encoded_dim = encoded_dim
        self.metric=metric
        self.optimizer = optimizers.Adamax() 
        self.model = self.build_model()
        self.model.compile(loss='mean_squared_error',optimizer=self.optimizer,metrics=[self.metric])
    def Encoder(self,x_test):
        model_Encoder    = Model(inputs=self.model.input, outputs=self.model.get_layer('Encoder').output)
        return model_Encoder.predict(x_test)
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.encoded_dim, input_dim=(self.inputDimentions),activation='linear',name="Encoder"))
        model.add(Dense(self.inputDimentions,activation='linear'))
        if (self.summary):
            model.summary()
        return model
    def train(self, epochs=200, batch_size=256,x_train=None,x_test=None):
        self.historyPca=self.model.fit(x=x_train, y=x_train,epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test),verbose=1)
        return self.historyPca
        
class Full_Conected_AE():
    def __init__(self,inputDimentions=784,encoded_dim=3,lossFunction="mean_squared_error",activation_encoder='linear',activation_decoder='linear',summary=False,metric='binary_crossentropy'):
        self.summary  = summary
        self.inputDimentions=inputDimentions
        self.encoded_dim = encoded_dim
        self.metric=metric
        self.activation_encoder=activation_encoder
        self.activation_decoder=activation_decoder
        self.optimizer = optimizers.Adamax() 
        self.loss=lossFunction
        self.model = self.build_model()
        self.model.compile(loss=self.loss,optimizer=self.optimizer,metrics=[self.metric])
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.encoded_dim, input_dim=(self.inputDimentions),activation=self.activation_encoder,name="Encoder"))
        model.add(Dense(self.inputDimentions,activation=self.activation_decoder,name="Decoder"))
        if (self.summary):
            model.summary()
        return model
    def Encoder(self,x_test):
        model_Encoder    = Model(inputs=self.model.input, outputs=self.model.get_layer('Encoder').output)
        return model_Encoder.predict(x_test)
    def Decoder(self,x_test):
        return self.model.predict(x_test)
    def train(self, epochs=200, batch_size=256,x_train=None,x_test=None):
        self.historyAE=self.model.fit(x=x_train, y=x_train,epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test),verbose=1)
        return self.historyAE

class Deep_AE():
    def __init__(self,inputDimentions=784,encoded_dim=3,lossFunction="mean_squared_error",activation_encoder='linear',activation_decoder='linear',summary=False,metric='binary_crossentropy'):
        self.summary  = summary
        self.inputDimentions=inputDimentions
        self.encoded_dim = encoded_dim
        self.encoded_dim1 = 128
        self.encoded_dim2 = 64
        self.metric=metric
        self.activation_encoder=activation_encoder
        self.activation_decoder=activation_decoder
        self.optimizer = optimizers.Adamax() 
        self.loss=lossFunction
        self.model = self.build_model()
        self.model.compile(loss=self.loss,optimizer=self.optimizer,metrics=[self.metric])
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.encoded_dim1, input_dim=(self.inputDimentions),activation=self.activation_encoder,name="Encoder_1"))
        model.add(Dense(self.encoded_dim2,activation=self.activation_encoder,name="Encoder_2"))
        model.add(Dense(self.encoded_dim  ,activation=self.activation_encoder,name="Encoder"))

        model.add(Dense(self.encoded_dim2,activation=self.activation_encoder,name="Decoder_2",input_dim=(self.encoded_dim)))
        model.add(Dense(self.encoded_dim1,activation=self.activation_encoder,name="Decoder_1"))
        model.add(Dense(self.inputDimentions,activation=self.activation_decoder,name="Decoder"))        
        if (self.summary):
            model.summary()
        return model
    def Encoder(self,x_test):
        model_Encoder    = Model(inputs=self.model.input, outputs=self.model.get_layer('Encoder').output)
        return model_Encoder.predict(x_test)
    def DecoderOnly(self,x_test):
        _decoder=[]
        for i, layer in enumerate(self.model.layers):   
            if (i>2) & (i<6):
                _decoder.append(layer)
        model_Decoder    =Sequential(_decoder)
        return model_Decoder.predict(x_test)
    def Decoder(self,x_test):
        return self.model.predict(x_test)
    def train(self, epochs=200, batch_size=256,x_train=None,x_test=None):
        self.historyAE=self.model.fit(x=x_train, y=x_train,epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test),verbose=1)
        return self.historyAE


class CAE():
    def __init__(self,inputDimentions=(28,28,1),summary=False):
        self.summary  = summary
        self.inputDimentions=inputDimentions
        self.encoded_dim1 = 128
        self.encoded_dim2 = 64
        self.optimizer = optimizers.Adamax() 
        self.model = self.build_model()
        self.model.compile(loss='binary_crossentropy',optimizer=self.optimizer,metrics=['binary_crossentropy'])
    def build_model(self):
        model_CAE = Sequential()
        model_CAE.add(Conv2D(16, (3,3),activation='relu',padding='same',input_shape=(28,28,1)))
        model_CAE.add(MaxPooling2D(pool_size=(2,2)))
        model_CAE.add(Conv2D(8, (3,3),activation='relu',padding='same'))
        model_CAE.add(MaxPooling2D((2,2)))

        model_CAE.add(Conv2D(8, (3,3),activation='relu',padding='same',name='Encoder'))

        model_CAE.add(UpSampling2D((2,2)))
        model_CAE.add(Conv2D(16, (3,3),activation='relu',padding='same'))
        model_CAE.add(UpSampling2D((2,2)))
        model_CAE.add(Conv2D(1, (3,3),activation='sigmoid',padding='same'))    
        if (self.summary):
            model_CAE.summary()
        return model_CAE
    def Encoder(self,x_test):
        model_Encoder    = Model(inputs=self.model.input, outputs=self.model.get_layer('Encoder').output)
        return model_Encoder.predict(x_test)
    def Decoder(self,x_test):
        return self.model.predict(x_test)
    def train(self, epochs=200, batch_size=256,x_train=None,x_test=None):
        self.historyAE=self.model.fit(x=x_train, y=x_train,epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test),verbose=1)
        return self.historyAE
