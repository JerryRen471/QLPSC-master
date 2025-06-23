import torch as tc


def gtn(para=dict()):
    para.update(program())
    para.update(lps())
    para.update(feature_map())
    para.update(ml())
    para.update(training())
    # Program Parameter
    para['dtype'] = tc.complex128
    para['device'] = 'cpu'
    # LPS Parameter
    para['retained_feature'] = 10
    para['virtual_bond'] = 16
    para['lps_rand_seed'] = 1
    # Machine Learning Parameter
    para['measure_train'] = 50
    para['sample_num'] = 1000
    para['batch'] = 50000
    para['rand_index_seed'] = 1
    # Training Parameter
    para['update_step'] = 2e-1
    para['step_decay_rate'] = 2  # 5
    para['step_accuracy'] = 1e-4  # 5e-3
    para['converge_type'] = 'cost function'
    para['converge_accuracy'] = 2e-5  # 1e-4
    # Path
    para['load_state_path'] = '/Users/wenjun/code/Data/target_state/Rand_large/chain%d/state%d_normal_1_3_0.pr' % (para['retained_feature'], para['retained_feature'])
    para['data_path'] = '/Users/wenjun/code/Data/Random_Glps/chain%d/normal_1_3_0/Nm_noCounter/' % (para['retained_feature'])
    para['save_result_path'] = '/Users/wenjun/code/Result/Random_Glps/chain%d/normal_1_3_0/' % (para['retained_feature'])
    return para


def program(para=dict()):
    para['dtype'] = tc.complex128
    para['device'] = 'cpu'
    return para


def lps(para=dict()):
    para['tensor_network_type'] = 'LPS'
    para['retained_feature'] = 10
    para['physical_bond'] = 2
    para['virtual_bond'] = 16
    para['purification_bond1'] = 1
    para['purification_bond2'] = 2
    para['tensor_initialize_type'] = 'rand'
    para['lps_rand_seed'] = 1
    return para


def feature_map(para=dict()):
    para['map_module'] = 'tomography'
    para['mapped_dimension'] = 2
    return para


def ml(para=dict()):
    para['dataset'] = 'LPS_tomography'
    para['measure_train'] = 100
    para['sample_num'] = 8192
    para['batch'] = 10000
    para['sort_module'] = 'rand'
    para['rand_index_seed'] = 1
    return para


def training(para=dict()):
    para['update_step'] = 2e-1
    para['step_decay_rate'] = 5
    para['step_accuracy'] = 5e-3
    para['converge_type'] = 'cost function'  # or 'fidelity'
    para['converge_accuracy'] = 1e-5
    return para