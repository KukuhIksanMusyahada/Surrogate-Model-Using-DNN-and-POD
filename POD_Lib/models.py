import os
import tensorflow as tf
import numpy as np
import importlib
import pandas as pd
import matplotlib.pyplot as plt
import pickle


from keras.models import Sequential
from keras.layers import Dense


def my_models(x_input, label, k):
    '''This function is the used to define the model construction.
    The construction will consist several layer with dense layer type.
    This construction is the recomendation but can be changed if needed'''
    model= Sequential([
        Dense(200, input_dim=2),
        Dense(600,activation='relu'),
        Dense(300,activation='relu'),
        Dense(500,activation='relu'), #the layer size and its neurons can be changed
        Dense(k), #do not change this
        ])
    optim = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss= 'mae',
                  optimizer='adam',
                  )

    history = model.fit(x_input,label.T, epochs=1000, verbose=0)
    return model,history

def MAPE(true,pred):
    '''This function will calculate the Mean Absolute Percentage Error'''
    return np.mean(abs((true-pred)/true)) *100

def POD_Train(x_input, y_input,Mach=None,Vf='Low',case='CD'):
    '''Train the models using POD method for all value of K.
    The training will take at around 10 minutes but can be different depended on device specs'''
    U,s,V = perform_SVD(matrix= y_input)
    k_max = U.shape[0]
    k = np.arange(1,(k_max),1)
    if case=='CD':
      multiplier = 1000
    else:
      multiplier = 1
    for num in k:
        U_hat= calc_U_hat(U, num)
        delta_hat = calc_delta_hat(U_hat, sol_mat= y_input)
        delta_hat_mul = delta_hat *multiplier
        model, hist = my_models(x_input,delta_hat_mul,num)
        if Vf.capitalize()=='Low':
          savemodel(model, hist,num,VF=Vf.capitalize(),Mach=Mach,case=case)
        elif Vf.capitalize()=='High':
          savemodel(model, hist,num,VF=Vf.capitalize(),case=case)
        else:
          print('Wrong Input VF Type!')

    return 

def savemodel(model, history,K,VF='Low',Mach='None',case='CD' ,model_path: str=Models):
    """Save both model and history"""
    path = os.path.join(model_path,case.capitalize())
    if VF.capitalize() == 'Low':
      path = os.path.join(path,'LowVF')
      if Mach != None:
        path= os.path.join(path,Mach)
    elif VF.capitalize()=='High':
      path = os.path.join(path,'HighVF')
    else:
      print('Only Accept "Low and High" for VF input!')


    folder_name = f'K{K}'
    model_directory= os.path.join(path,folder_name)
    if not os.path.exists(model_directory):
      os.makedirs(model_directory)
    else:
      folder_name=folder_name+'_1'
      model_directory= os.path.join(path,folder_name)
      os.makedirs(model_directory)
    history_file = os.path.join(model_directory, 'history.pkl')


    model.save(model_directory)
    print ("\nModel saved to {}".format(model_directory))

    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)
    print ("Model history saved to {}".format(history_file))


def load_model(path,VF, num_model,Mach=None):
    """Load Model and optionally it's history as well"""
    if VF=='Low':
      path=os.path.join(path,'LowVF',Mach)
    elif VF=='High':
      path=os.path.join(path,'HighVF')
    else:
      return 'Model type is wrongly declared!'

    folder_name = 'K'+ str(num_model)

    path_to_model = os.path.join(path,folder_name)
    history_file = os.path.join(path_to_model, 'history.pkl')
    model = tf.keras.models.load_model(path_to_model)
    # model = tf.saved_model.load(path_to_model)
    print ("\nmodel loaded")

    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    print ("model history loaded")

    return model, history