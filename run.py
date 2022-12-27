from POD_Lib import calculation as calc
from POD_Lib import path_handling as ph
from POD_Lib import utility as ut
from POD_Lib import models as mod
from POD_Lib import matrix_pencil as mp



if __name__=="__main__":
    # Take Solution Matrix of mach 0.6
    in_cl1,sol_cl1,val1,test1 = calc.sol_matrix(tp=4,num_val=64,names='CL')
    in_cl2,sol_cl2,val2,test2 = calc.sol_matrix(tp=4,names='CL')
    in_cl3,sol_cl3,val3,test3 = calc.sol_matrix(path=ph.get_raw_data(),tp=4,names='CL')
    in_cd1,sol_cd1,val1,test1 = calc.sol_matrix(tp=4,num_val=64,names='CD')
    in_cd2,sol_cd2,val2,test2 = calc.sol_matrix(tp=4,names='CD')
    in_cd3,sol_cd3,val3,test3 = calc.sol_matrix(path=ph.get_raw_data(),tp=4,names='CD')
    in_pi1,sol_pi1,val1,test1 = calc.sol_matrix(tp=4,num_val=64,names='Pitch')
    in_pi2,sol_pi2,val2,test2 = calc.sol_matrix(tp=4,names='Pitch')
    in_pi3,sol_pi3,val3,test3 = calc.sol_matrix(path=ph.get_raw_data(),tp=4,names='Pitch')
    in_pl1,sol_pl1,val1,test1 = calc.sol_matrix(tp=4,num_val=64,names='Plunge')
    in_pl2,sol_pl2,val2,test2 = calc.sol_matrix(tp=4,names='Plunge')
    in_pl3,sol_pl3,val3,test3 = calc.sol_matrix(path=ph.get_raw_data(),tp=4,names='Plunge')
    # Load Models POD
    model_cd1, hist_cd1 = mod.load_model(case='POD',names='CD',num_model=40,sample='1')
    model_cd2, hist_cd2 = mod.load_model(case='POD',names='CD',num_model=40,sample='2')
    model_cd3, hist_cd3 = mod.load_model(case='POD',names='CD',num_model=40,sample='3')

    model_cl1, hist_cl1 = mod.load_model(case='POD',names='CL',num_model=40,sample='1')
    model_cl2, hist_cl2 = mod.load_model(case='POD',names='CL',num_model=40,sample='2')
    model_cl3, hist_cl3 = mod.load_model(case='POD',names='CL',num_model=40,sample='3')

    model_pi1, hist_pi1 = mod.load_model(case='POD',names='Pitch',num_model=40,sample='1')
    model_pi2, hist_pi2 = mod.load_model(case='POD',names='Pitch',num_model=40,sample='2')
    model_pi3, hist_pi3 = mod.load_model(case='POD',names='Pitch',num_model=40,sample='3')

    model_pl1, hist_pl1 = mod.load_model(case='POD',names='Plunge',num_model=40,sample='1')
    model_pl2, hist_pl2 = mod.load_model(case='POD',names='Plunge',num_model=40,sample='2')
    model_pl3, hist_pl3 = mod.load_model(case='POD',names='Plunge',num_model=40,sample='3')
    # Load Models Multiplier
    # Multiplier Models
    model1, hist1 = mod.load_model(case='Non_POD',names='CD')
    model2, hist2 = mod.load_model(case='Non_POD',names='CL')
    model3, hist3 = mod.load_model(case='Non_POD',names='Pitch')
    model4, hist4 = mod.load_model(case='Non_POD',names='Plunge')
