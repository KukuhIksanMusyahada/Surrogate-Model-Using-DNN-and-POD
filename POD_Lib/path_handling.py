import os

def get_this_dir():
    return os.path.dirname( os.path.abspath(__file__) )


def get_data_source():
    return os.path.join(get_this_dir(), os.pardir, 'Data_Source')


def get_raw_data():
    return os.path.join(get_data_source(), 'Raw')

def M0_6():
    return os.path.join(get_raw_data(), 'M_0.6')

def M0_7():
    return os.path.join(get_raw_data(), 'M_0.7')

def M0_8():
    return os.path.join(get_raw_data(), 'M_0.8')


def get_result_data():
    return os.path.join(get_data_source(), 'Results')

def get_models():
    return os.path.join(get_data_source(),'Models')


