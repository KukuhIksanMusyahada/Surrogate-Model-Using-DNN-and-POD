import os

import numpy as np
import pandas as pd
import scipy

from scipy import linalg
from POD_Lib import utility as ut

# Define Solution Matrix
def sol_matrix(path,names='CD'):
    '''Solution Matrix is a matrix consist of target values, 
    the size of this matrix is m*n with m is the timestep and 
    n is the size of our dataset. The input of this function is
    only the path and the target features'''
    sol_mat = list()
    data_input = list()
    files = list()
    if names.upper() == 'PLUNGE':
        col = [['plunge(airfoil)'], ['plunge_airfoil']]

    if names.upper() == 'PITCH':
        col = [['pitch(airfoil)'], ['pitch_airfoil']]
        tp = 3
      

    
    for file in os.listdir(path):
        if file.endswith('.csv'):
            file_path = os.path.join(path, file)
            if  names.upper() == 'PLUNGE' or names.upper() == 'PITCH':
                try:
                    df = pd.read_csv(file_path, usecols= col[0], nrows= 114,engine='python').to_numpy()
                except ValueError:
                    df = pd.read_csv(file_path, usecols= col[1], nrows= 114,engine='python').to_numpy()
            else:
                df = pd.read_csv(file_path, usecols=[names.upper()], nrows= 114, engine='python').to_numpy()
            if np.isnan(df).any():
                continue
            grad = ut.gradien(df)
            turn_point = ut.find_turn_point(grad)
                # print(file)
            if len(turn_point) < tp:
                print(f'{file} has {len(turn_point)} turn point')
                continue
            files.append(file)
           
    np.random.seed(45)
    validation = np.random.choice(files, 4, replace=False)
    # print(validation)
    for file in files:
        if file not in validation:
            file_path = os.path.join(path, file)
            if  names.upper() == 'PLUNGE' or names.upper() == 'PITCH':
                try:
                    df = pd.read_csv(file_path, usecols= col[0], nrows= 114,engine='python').to_numpy()
                except ValueError:
                    df = pd.read_csv(file_path, usecols= col[1], nrows= 114,engine='python').to_numpy()
            else:
                df = pd.read_csv(file_path, usecols=[names.upper()], nrows= 114, engine='python').to_numpy()
            sol_mat.append(df)
            data_input.append(list(ut.extract_mach_and_vf(file)))
    sol_mat = np.concatenate(sol_mat, axis=1)
    
    return  np.array(data_input),sol_mat, validation

def perform_svd(matrix):
    U,s, V = linalg.svd(matrix, full_matrices= True)
    return U,s,V

def calc_u_hat(matrix, k):
    return matrix[:,:k]


def calc_delta_hat(matrix, sol_mat):
    return np.matmul(matrix.T, sol_mat)


def prediction(delta_star, u_hat):
    return np.matmul(u_hat, delta_star)