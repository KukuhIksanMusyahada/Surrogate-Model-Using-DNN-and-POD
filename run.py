from POD_Lib import calculation as calc
from POD_Lib import path_handling as ph
from POD_Lib import utility as ut
from POD_Lib import models as mod
from POD_Lib import matrix_pencil as mp



if __name__=="__main__":
    # Take Solution Matrix of mach 0.6
    X_Cd6,y_Cd6,val_Cd6 = calc.sol_matrix(ph.M0_6(),names='CD')
    X_Cl6,y_Cl6,val_Cl6 = calc.sol_matrix(ph.M0_6(),names='CL')
    X_Pl6,y_Pl6,val_Pl6 = calc.sol_matrix(ph.M0_6(),names='Plunge')
    X_Pi6,y_Pi6,val_Pi6 = calc.sol_matrix(ph.M0_6(),names='Pitch')
    # Take Solution Matrix of mach 0.7
    X_Cd7,y_Cd7,val_Cd7 = calc.sol_matrix(ph.M0_7(),names='CD')
    X_Cl7,y_Cl7,val_Cl7 = calc.sol_matrix(ph.M0_7(),names='CL')
    X_Pl7,y_Pl7,val_Pl7 = calc.sol_matrix(ph.M0_7(),names='Plunge')
    X_Pi7,y_Pi7,val_Pi7 = calc.sol_matrix(ph.M0_7(),names='Pitch')
    # Take Solution Matrix of mach 0.6
    X_Cd8,y_Cd8,val_Cd8 = calc.sol_matrix(ph.M0_8(),names='CD')
    X_Cl8,y_Cl8,val_Cl8 = calc.sol_matrix(ph.M0_8(),names='CL')
    X_Pl8,y_Pl8,val_Pl8 = calc.sol_matrix(ph.M0_8(),names='Plunge')
    X_Pi8,y_Pi8,val_Pi8 = calc.sol_matrix(ph.M0_8(),names='Pitch')
    # print(X_Cd6.shape,X_Cd7.shape,X_Cd8.shape,X_Cl6.shape,X_Cl7.shape,X_Cl8.shape)
    # print(X_Pl6.shape,X_Pl7.shape,X_Pl8.shape,X_Pi6.shape,X_Pi7.shape,X_Pi8.shape)
    #Trim Outliers the default has minimum threshold (-0.2) and maximum threshold 0.4
    # CD (doesnt need any trimming but if you feel like it, uncomment this codes below)
    # X_Cd6,y_Cd6 = ut.outlier_cleaner(X_Cd6,y_Cd6)
    # X_Cd7,y_Cd7 = ut.outlier_cleaner(X_Cd7,y_Cd7)
    # X_Cd8,y_Cd8 = ut.outlier_cleaner(X_Cd8,y_Cd8)

    # CL (doesnt need any trimming but if you feel like it, uncomment this codes below)
    # X_Cl6,y_Cl6 = ut.outlier_cleaner(X_Cl6,y_Cl6)
    # X_Cl7,y_Cl7 = ut.outlier_cleaner(X_Cl7,y_Cl7)
    # X_Cl8,y_Cl8 = ut.outlier_cleaner(X_Cl8,y_Cl8)

    #Plunge
    X_Pl7,y_Pl7 = ut.outlier_cleaner(X_Pl7,y_Pl7,max_thres=0.05,min_thres=-0.05)

    #Pitch
    X_Pi8,y_Pi8= ut.outlier_cleaner(X_Pi8,y_Pi8,max_thres=0.05,min_thres=-0.05)
    print('Do you want to train or retrain the model?\n This may take a long time! \n')
    is_train = input( '\t 1. Yes \n \t 2. No\n')
    if is_train == '1':
        print('Training CD')
        mod.pod_train(X_Cd6,y_Cd6,Mach='M0_6',case='CD')
        mod.pod_train(X_Cd7,y_Cd7,Mach='M0_7',case='CD')
        mod.pod_train(X_Cd8,y_Cd8,Mach='M0_8',case='CD')
        print('Training CL')
        mod.pod_train(X_Cl6,y_Cl6,Mach='M0_6',case='CL')
        mod.pod_train(X_Cl7,y_Cl7,Mach='M0_7',case='CL')
        mod.pod_train(X_Cl8,y_Cl8,Mach='M0_8',case='CL')
        print('Training Pitch')
        mod.pod_train(X_Pi6,y_Pi6,Mach='M0_6',case='Pitch')
        mod.pod_train(X_Pi7,y_Pi7,Mach='M0_7',case='Pitch')
        mod.pod_train(X_Pi8,y_Pi8,Mach='M0_8',case='Pitch')
        print('Training CD')
        mod.pod_train(X_Pl6,y_Pl6,Mach='M0_6',case='Plunge')
        mod.pod_train(X_Pl7,y_Pl7,Mach='M0_7',case='Plunge')
        mod.pod_train(X_Pl8,y_Pl8,Mach='M0_8',case='Plunge')
    else:
        print('Input other than "1" is ignored and treated as "Do not train/retrain the model!"')
    Mach = float(input('Please Enter Mach Number!\n'))
    Vf = float(input('Please Enter Flutter Speed Index!\n'))
    print('The best K value to predict lie within range 20-60')
    K = int(input('Please Enter The K value (how many label needed will be predicted)\n'))

    # Predict CD
    if Mach>=0.6 and Mach <0.7:
        X,y = X_Cd6,y_Cd6
    elif Mach>=0.7 and Mach <0.8:
        X,y = X_Cd7,y_Cd7
    elif Mach>=0.8 and Mach <0.9:
        X,y = X_Cd8,y_Cd8
    else:
        print('This Model is only designed to predict the aeroelastic properties within rangr 0.6-0.9!') 
    mod.POD_Predict(X,y,Mach,Vf,K,case='CD',Vf_type='Low')

    # Predict CL
    if Mach>=0.6 and Mach <0.7:
        X,y = X_Cl6,y_Cl6
    elif Mach>=0.7 and Mach <0.8:
        X,y = X_Cl7,y_Cl7
    elif Mach>=0.8 and Mach <0.9:
        X,y = X_Cl8,y_Cl8
    else:
        print('This Model is only designed to predict the aeroelastic properties within rangr 0.6-0.9!') 
    mod.POD_Predict(X,y,Mach,Vf,K,case='CL',Vf_type='Low')
    # Predict Pitch
    if Mach>=0.6 and Mach <0.7:
        X,y = X_Pi6,y_Pi6
    elif Mach>=0.7 and Mach <0.8:
        X,y = X_Pi7,y_Pi7
    elif Mach>=0.8 and Mach <0.9:
        X,y = X_Pi8,y_Pi8
    else:
        print('This Model is only designed to predict the aeroelastic properties within rangr 0.6-0.9!') 
    Pitch = mod.POD_Predict(X,y,Mach,Vf,K,case='Pitch',Vf_type='Low')
    # Predict Plunge
    if Mach>=0.6 and Mach <0.7:
        X,y = X_Pl6,y_Pl6
    elif Mach>=0.7 and Mach <0.8:
        X,y = X_Pl7,y_Pl7
    elif Mach>=0.8 and Mach <0.9:
        X,y = X_Pl8,y_Pl8
    else:
        print('This Model is only designed to predict the aeroelastic properties within rangr 0.6-0.9!') 
    Plunge = mod.POD_Predict(X,y,Mach,Vf,K,case='Plunge',Vf_type='Low')        
    
    damping = mp.compute_damping_coefficient(Pitch,Plunge)
    print(f'The damping coefficient of {Mach} and {Vf} is {damping}')