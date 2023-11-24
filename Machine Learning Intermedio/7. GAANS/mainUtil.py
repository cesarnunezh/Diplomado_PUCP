import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
from keras import backend as K
from keras.layers import Input, Dense,Conv2D, Conv2DTranspose, Lambda, Flatten, Reshape
from keras.models import model_from_json
from keras.datasets import mnist
#%matplotlib inline

def LoadMnist(isVector=False):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Transformando os dados a formato "float32"
    X_train   =   X_train.astype('float32')/255.
    X_test    =   X_test.astype('float32')/255.
    if isVector :
        X_train   = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
        X_test    = X_test.reshape((len(X_test)  , np.prod(X_test.shape[1:] )))
    return (X_train, y_train, X_test, y_test)

def PlotDataAE(X,X_AE,digit_size=28,cmap='jet',Only_Result=True):
    plt.figure(figsize=(digit_size,digit_size))
    if (Only_Result):
        for i in range(0,20):
            plt.subplot(10,10,(i+1))
            plt.imshow(np.squeeze(X[i].reshape(digit_size,digit_size,1)),cmap=cmap)
            plt.axis('off')
            plt.title('Input')
        plt.figure(figsize=(digit_size,digit_size))
    for i in range(0,20):
        plt.subplot(10,10,(i+1))
        plt.imshow(np.squeeze(X_AE[i].reshape(digit_size,digit_size,1)),cmap=cmap) 
        plt.axis('off')
        plt.title('Output')
    plt.show()

def PlotHistory(history):
    leg=[]
    for key in history.keys():
        plt.plot(history[key])
        plt.title('Training')
        plt.xlabel('epoch') 
        leg.append(key)
        #print(key,"  : ",history[key][-5:-1])
    plt.legend(leg, loc='upper left')

def PlotMeanStd(x_encoded,b=10):
    plt.subplot(1,2,1)
    plt.hist(np.mean(x_encoded,axis=1),bins=b)
    plt.hist(np.std(x_encoded,axis=1),bins=b)
    plt.subplot(1,2,2)
    plt.hist(x_encoded[np.random.randint(len(x_encoded))],bins=5*b)
    plt.show()

def Save_Model(modelo,name):
    def save(model, model_name):
        model_path = "%s.json" % model_name
        weights_path = "%s_weights.hdf5" % model_name
        options = {"file_arch": model_path, 
                   "file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])
    save(modelo,name)
def Load_Model(name):
    def load(model_name):
        model_path = "%s.json" % model_name
        weights_path = "%s_weights.hdf5" % model_name
        # load json and create model
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()      
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(weights_path)        
        return loaded_model
    modelo=load(name)
    return modelo

def LoadMPS45(dirBase='/work/ProyectosSmithDL/BaseDeDatos/MPS45.mat',AllTrain=False):
    if K.image_data_format() == 'channels_first':
        original_img_size = (1, 45, 45)
    else:
        original_img_size = (45, 45, 1)
    EnsIni= sio.loadmat(dirBase)
    x_Facies=np.transpose(EnsIni['Dato']).astype('float32')
    
    if AllTrain :
        x_train =x_Facies
    else:
        x_train =x_Facies[0:25000]

    x_test  =x_Facies[25000:29997]
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train = x_train.reshape((x_train.shape[0],) + (original_img_size))
    x_test =  x_test.reshape((x_test.shape[0],) + (original_img_size))
    return x_train,x_test
def LoadMPS100(dirBase='/work/ProyectosSmithDL/BaseDeDatos/MPS100.mat',AllTrain=False):
    if K.image_data_format() == 'channels_first':
        original_img_size = (1, 100, 100)
    else:
        original_img_size = (100, 100, 1)
    x_Facies = {}
    f = h5py.File('BaseMPS100(1).mat')
    for k, v in f.items():
        x_Facies[k] = np.array(v)      
    x_Facies=x_Facies['Dato'].astype('float32')
    f.close()
    if AllTrain :
        x_train =x_Facies
    else :
        x_train =x_Facies[0:32000]
    x_test  =x_Facies[32000:40000]
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train = x_train.reshape((x_train.shape[0],) + (original_img_size))
    x_test =  x_test.reshape((x_test.shape[0],) + (original_img_size))
    return x_train,x_test
