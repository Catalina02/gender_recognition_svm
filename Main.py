
'''IMPORTACION DE MODULOS'''
# Ignore  the warnings
from typing import ValuesView
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns 
import missingno as msno

#configure 
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)
import os

#import the necessary modelling algos.
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

#preprocess.
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer

#audio
import pyaudio
import wave

#Cargar  DataFrame
train=pd.read_csv(r'training_data.csv')
df=train.copy()

#Elimina Columnas Innecesarias o Redundantes
def Eliminar_Columnas():
    df.drop('centroid',axis=1,inplace=True)
    df.drop('Q75',axis=1,inplace=True)
    df.drop('skew',axis=1,inplace=True)
    df.drop('kurt',axis=1,inplace=True)
    df.drop('mode',axis=1,inplace=True)
    df.drop('minfun',axis=1,inplace=True)
    df.drop('maxfun',axis=1,inplace=True)
    df.drop('meandom',axis=1,inplace=True)
    df.drop('mindom',axis=1,inplace=True)
    df.drop('maxdom',axis=1,inplace=True)
    df.drop('dfrange',axis=1,inplace=True)
    df.drop('modindx',axis=1,inplace=True)
    df.drop('sfm',axis=1,inplace=True)
    df.drop('sp.ent',axis=1,inplace=True)
    df.drop('meanfreq',axis=1,inplace=True)
    df.drop('median',axis=1,inplace=True)
    df.drop('sd',axis=1,inplace=True)
    df.drop('IQR',axis=1,inplace=True)

#Caracteristicas segun Genero
def Car_Genero(feature):
    sns.factorplot(data=df,y=feature,x='label',kind='strip',palette='YlGnBu')
    plt.title(feature.upper())
    fig=plt.gcf()
    fig.set_size_inches(7,7)
    fig.savefig(feature.upper()+'caract-gen') #guardar graficos 

def calcular_limites(feature): 
    q1,q3=df[feature].quantile([0.25,0.75])  #q1->primercuartil->Q25; q3->tercercuartil->Q75
    iqr=q3-q1#rango intercuartil=iferencia entre el tercer y el primer cuartil, Mediante esta medida se eliminan los valores extremadamente alejados.
    #En una distribución, encontramos la mitad de los datos, el 50 %, ubicados dentro del rango intercuartílico. 
    rang=1.5*iqr
    #Q25-RANG= Limite Inferior; Q75+RANG=Limite Superior
    return(q1-rang,q3+rang)             

def Normalizar():
        temp_df=df
        scaler=StandardScaler()
        scaled_df=scaler.fit_transform(temp_df.drop('label',axis=1))
        X=scaled_df
        Y=df['label'].to_numpy()
        x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42) 
        #test_size=proporción del conjunto de datos para incluir en la división de prueba. 
        #random_state=Controla la mezcla aplicada a los datos antes de aplicar la división. Pase un int para una salida reproducible a través de múltiples llamadas a funciones 
        return x_train,x_test,y_train,y_test,scaler

def Modelado(modelo):
        Normalizar()
        clf_svm=modelo 
        clf_svm.fit(x_train,y_train)
        pred=clf_svm.predict(x_test)
        print(accuracy_score(pred,y_test))
        return clf_svm

def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "test_audio.wav"

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("* recording")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def Obtener_Valores_Voz():
        
        os.system('"Praat.exe" --run extract_freq_info.praat')
        file = open('output.txt','r') 
        values = file.readline()
        values = values.split(', ')
        values=values[0:2]
        for x in range(0,2):
                values[x] = float(values[x])/1000
        values=np.array(values)
        valores_voz=[values]
        return valores_voz

#************   MAIN   ******************


for col in (df.columns):
    if (col != 'label'):
        lower,upper=calcular_limites(col)
        df = df[(df[col] >lower) & (df[col]<upper)]     

Eliminar_Columnas()

x_train,x_test,y_train,y_test,scaler=Normalizar()
clf=Modelado(SVC(kernel='rbf', C=1,gamma=0.1))

#record_audio()
voz=Obtener_Valores_Voz()
clf.predict(scaler.transform(voz))



'''TESTEOS'''

dict = {'label':{'male':0,'female':1}}      # label = column name
df.replace(dict,inplace = True)           # replace = str to numerical
x = df.loc[:, df.columns != 'label']
y = df.loc[:,'label']

X_train,X_test,Y_train,Y_test,Scaler=x_train,x_test,y_train,y_test,scaler


classifier = SVC(kernel='rbf', C=1,gamma=0.1, probability=True)
classifier.fit(X_train, Y_train)

def GraficosModelo(X_train,X_test,Y_train,Y_test,classifier):
    fig = plt.figure(figsize=(9, 8))

    ax = fig.add_subplot(2, 2, 1)
    Y_pred = classifier.predict(X_test)
    clr=np.asarray(['b','g'])
    ax.scatter(X_test[Y_test==0, 0], X_test[Y_test==0, 1], c=clr[Y_pred[Y_test==0]], 
            marker='x', cmap=plt.cm.RdBu, vmin=0, vmax=1,
            linewidth=1, alpha=0.5, s=40, label='Clase 1')
    ax.scatter(X_test[Y_test==1, 0], X_test[Y_test==1, 1], c=clr[Y_pred[Y_test==1]], 
            marker='o', cmap=plt.cm.RdBu, vmin=0, vmax=1, 
            linewidth=0, alpha=0.5, s=40, label='Clase 2')
    plt.title('Clasificaci\'on en el conjunto de test')

    ax = fig.add_subplot(2, 2, 2)
    Y_pred = classifier.predict_proba(X_test)
    fpr, tpr, th = roc_curve(Y_test, Y_pred[:, 1])
    ax.plot(fpr, tpr, linewidth=4, alpha=0.5, label='Test')
    print("Area bajo la curva ROC (test): %f" %(auc(fpr, tpr)))
    fpr, tpr, th = roc_curve(Y_train, classifier.predict_proba(X_train)[:, 1])
    ax.plot(fpr, tpr, linewidth=4, alpha=0.5, label='Train')
    print("Area bajo la curva ROC (train): %f" %(auc(fpr, tpr)))
    plt.legend(loc=4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('curva ROC')
    plt.grid()

    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(X_train[Y_train==0, 0], X_train[Y_train==0, 1], c='b', 
            marker='x', linewidth=1, alpha=0.5, s=20, label='Clase 1')
    ax.scatter(X_train[Y_train==1, 0], X_train[Y_train==1, 1], c='g', 
            marker='o', linewidth=0, alpha=0.5, s=20, label='Clase 2')
    ax.scatter(X_train[classifier.support_, 0], X_train[classifier.support_, 1], 
            c='r', linewidth=1, alpha=0.25, s=100)
    plt.title('Vectores de soporte')

    plt.tight_layout()
    print("%d SVs para la clase 1" % (classifier.n_support_[0]))
    print("%d SVs para la clase 2" % (classifier.n_support_[1]))
    
    
GraficosModelo(x_train,x_test,y_train,y_test,classifier)