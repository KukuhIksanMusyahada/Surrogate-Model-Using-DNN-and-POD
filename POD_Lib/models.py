import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


from keras.models import Sequential
from keras.layers import Dense

from POD_Lib import utility as ut
from POD_Lib import calculation as calc
from POD_Lib import path_handling as ph


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

def mape(true,pred):
    '''This function will calculate the Mean Absolute Percentage Error'''
    return np.mean(abs((true-pred)/true)) *100

def pod_train(x_input, y_input,Mach=None,Vf='Low',case='CD'):
    '''Train the models using POD method for all value of K.
    The training will take at around 10 minutes but can be different depended on device specs'''
    U,s,_ = calc.perform_svd(matrix= y_input)
    k_max = U.shape[0]
    k = np.arange(1,(k_max),1)
    if case=='CD':
      multiplier = 1000
    else:
      multiplier = 1
    for num in k:
        U_hat= calc.calc_u_hat(U, num)
        delta_hat = calc.calc_delta_hat(U_hat, sol_mat= y_input)
        delta_hat_mul = delta_hat *multiplier
        model, hist = my_models(x_input,delta_hat_mul,num)
        if Vf.capitalize()=='Low':
          save_model(model, hist,num,VF=Vf.capitalize(),Mach=Mach,case=case)
        elif Vf.capitalize()=='High':
          save_model(model, hist,num,VF=Vf.capitalize(),case=case)
        else:
          print('Wrong Input VF Type!')

    return 

def save_model(model, history,K,VF='Low',Mach='None',case='CD' ,model_path: str=ph.get_models()):
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

def POD_Validate(x_input, y_input,mach, vf,Vf_type=None,case='CD',path = ph.get_models()):
  U,s,_ = calc.perform_svd(matrix= y_input)
  k_max = U.shape[0]
  k = np.arange(1,(k_max),1)
  file_name = f'M_{str(mach)}_VF_{str(vf)}.csv'
  if case.upper()=='CD':
    multiplier=1000
  else:
    multiplier=1
  if case.upper() == 'PLUNGE':
    col = [['plunge(airfoil)'], ['plunge_airfoil']]
  if case.upper() == 'PITCH':
      col = [['pitch(airfoil)'], ['pitch_airfoil']]
  if mach < 0.7:
        file_path = os.path.join(ph.M0_6(),file_name)
        Mach='M0_6'
  elif mach >= 0.7 and mach < 0.8:
      file_path = os.path.join(ph.M0_7(),file_name)
      Mach='M0_7'
  elif mach >= 0.8:
      file_path = os.path.join(ph.M0_8(),file_name)
      Mach='M0_8'
  if  case.upper() == 'PLUNGE' or case.upper() == 'PITCH':
      try:
          file = pd.read_csv(file_path, usecols= col[0], nrows= 114,engine='python').to_numpy()
      except ValueError:
          file = pd.read_csv(file_path, usecols= col[1], nrows= 114,engine='python').to_numpy()
  else:
      file = pd.read_csv(file_path, usecols=[case.upper()], nrows= 114, engine='python').to_numpy()
  
  Mape = []
  path = os.path.join(path,case.capitalize())
  if Vf_type==None:
    if vf>1.4:
      VF = 'High'
    else:
      VF = 'Low'
  else:
    VF=Vf_type
  
  for num in k:

    U_hat= calc.calc_u_hat(U, num)

    if VF == 'Low':
      model, _ = load_model(path,VF,num,Mach)
    else:
      model, _ = load_model(path,VF,num)
    res= (model.predict([[mach,vf]])).T/multiplier
    predict = calc.prediction(res, U_hat)
    mape = mape(file,predict)
    Mape.append(mape)
    print(f'MAPE with k={num} is {mape}')
    
    plt.plot(predict, label=f'prediction at k={num}',)
    plt.plot(file, label='data')
    plt.title(f'{case.upper()} Curve of M_{str(mach)}_VF_{str(vf)}')
    plt.xlabel('Time Step')
    plt.ylabel(case.upper())
    plt.legend()
    plt.show()
  print(f'The minimum MAPE is on k={np.argmin(Mape)+1} with the value {np.min(Mape)}')
  return np.array(Mape)



def POD_Predict(x_input, y_input,mach, vf,k,case='CD',Vf_type=None,path = ph.get_models()):
  '''Predict a new data'''
  U,s,_ = calc.perform_svd(matrix= y_input)
  file_name = f'M_{str(mach)}_VF_{str(vf)}.csv'
  if case.upper() == 'PLUNGE':
    col = [['plunge(airfoil)'], ['plunge_airfoil']]
  if case.upper() == 'PITCH':
      col = [['pitch(airfoil)'], ['pitch_airfoil']]
  if mach < 0.7:
        file_path = os.path.join(ph.M0_6(),file_name)
        Mach='M0_6'
  elif mach >= 0.7 and mach < 0.8:
      file_path = os.path.join(ph.M0_7(),file_name)
      Mach='M0_7'
  elif mach >= 0.8:
      file_path = os.path.join(ph.M0_8(),file_name)
      Mach='M0_8'
  if  case.upper() == 'PLUNGE' or case.upper() == 'PITCH':
      try:
          file = pd.read_csv(file_path, usecols= col[0], nrows= 114,engine='python').to_numpy()
      except ValueError:
          file = pd.read_csv(file_path, usecols= col[1], nrows= 114,engine='python').to_numpy()
  else:
      file = pd.read_csv(file_path, usecols=[case.upper()], nrows= 114, engine='python').to_numpy()
  
  Mape = []
  path = os.path.join(path,case.capitalize())
  if Vf_type==None:
    if vf>1.4:
      VF = 'High'
    else:
      VF = 'Low'
  else:
    VF=Vf_type

  U_hat= calc.calc_u_hat(U, k)

  if VF == 'Low':
    model, _ = load_model(path,VF,k,Mach)
  else:
    model, _ = load_model(path,VF,k)
  if case =='CD':
    multiplier=1000
  elif case=='CL':
    multiplier=10
  else:
    multiplier=1
  
  res= (model.predict([[mach,vf]])).T/multiplier
  predict = calc.prediction(res, U_hat)
  mape = mape(file,predict)
  Mape.append(mape)
  print(f'MAPE with k={k} is {mape}')
  
  plt.plot(predict, label=f'prediction at k={k}',)
  plt.plot(file, label='data')
  plt.title(f'{case} Curve of M_{str(mach)}_VF_{str(vf)}')
  plt.xlabel('Time Step')
  plt.ylabel(f'{case}')
  plt.legend()
  plt.show()
