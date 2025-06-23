import numpy as np
import torch as tc
from library import Parameters
import math
import copy
import time


class MachineLearning:

    def __init__(self, para=Parameters.ml()):
        # Initialize Parameters
        self.para = copy.deepcopy(para)
        self.input_data = dict()
        self.data_info = dict()
        self.tensor_input = tuple()
        self.update_info = dict()
        self.tmp = dict()

    def initialize_dataset(self):
        self.load_dataset()
        self.arrange_dataset()
        self.data_info['n_training'], self.data_info['n_feature'] = self.input_data['sorted_train'].shape
        print('Size of the original dataset is %s' % str(self.input_data['sorted_train'].shape))

    def load_dataset(self):
        if self.para['dataset'] == 'LPS_tomography':
            xy = np.loadtxt(self.para['data_path'] + '%d_%d.txt' % (self.para['measure_train'], self.para['sample_num']), delimiter=',', dtype=np.float64, skiprows=1)
            f = open(self.para['data_path'] + '%d_%d.txt' % (self.para['measure_train'],self.para['sample_num']))
            train_test_num = list(range(1))  # 读取第一行
            for i in range(1):
                line = f.readline().strip()
                data_list = []
                num = list(map(float, line.split()))
                data_list.append(num)
                train_test_num[i] = int(np.array(data_list))
            train_num = train_test_num[0]
            self.input_data['train'] = tc.from_numpy(xy[0:train_num, 0:])

    def arrange_dataset(self):
        if self.para['sort_module'] == 'rand':
            self.input_data['sorted_train'] = self.rand_sort_data(self.input_data['train'])

    def rand_sort_data(self, input_data):
        np.random.seed(self.para['rand_index_seed'])
        rand_index = np.random.permutation(input_data.shape[0])
        input_data_rand_sorted = input_data[rand_index, :]
        return input_data_rand_sorted

    def feature_map(self, data_mapping):
        if self.para['map_module'] == 'tomography':
            # 定义常量
            a = 1 / math.sqrt(2)
            b = tc.tensor(1j / math.sqrt(2), dtype=tc.complex128)
            # 创建一个映射表
            mapping_table = tc.tensor([
                [a, a],   # 0
                [a, -a],  # 1
                [a, b],   # 2
                [a, -b],  # 3
                [1, 0],   # 4
                [0, 1],   # 5
                ], device=self.para['device'], dtype=self.para['dtype'])
            data_mapping = tc.tensor(data_mapping, dtype=tc.long)
            data_mapped = mapping_table[data_mapping]  # 直接索引映射表，得到结果
        else:
            data_mapped = False
        return data_mapped

    def generate_update_info(self):
        self.update_info['is_converged'] = 'untrained'
        self.update_info['update_position'] = 'unknown'
        self.update_info['update_direction'] = +1
        self.update_info['step'] = self.para['update_step']
        self.update_info['batch'] = self.para['batch']
        self.update_info['epochs_learned'] = 0
        self.update_info['cost_function_epochs'] = list()
        self.update_info['fidelity_yang_epochs'] = list()
        self.update_info['fidelity_epochs'] = list()

    def calculate_running_time(self, mode='end'):
        if mode == 'start':
            self.tmp['start_time'] = time.time()
        elif mode == 'end':
            self.tmp['end_time'] = time.time()

    def print_running_time(self):
        print('This epoch consumes ' + str(self.tmp['end_time'] - self.tmp['start_time']) + ' seconds.')

    def is_converge(self):
        epochs_learned = self.update_info['epochs_learned']
        cost_function_epochs = self.update_info['cost_function_epochs']
        fidelity_epochs = self.update_info['fidelity_yang_epochs']
        if self.para['converge_type'] == 'cost function':
            self.update_info['is_converged'] = bool(
                abs((cost_function_epochs[epochs_learned - 1] - cost_function_epochs[epochs_learned])
                    /cost_function_epochs[epochs_learned - 1]) < (self.para['converge_accuracy']) / self.update_info['step'])  # 1e-5/2e-1
            if self.update_info['is_converged']:
                if self.update_info['step'] > self.para['step_accuracy']:   # 5e-3
                    self.update_info['step'] /= self.para['step_decay_rate']   # 5
                    print('update step reduces to ' + str(self.update_info['step']))
                    self.update_info['is_converged'] = False
        elif self.para['converge_type'] == 'fidelity':
            self.update_info['is_converged'] = bool(
                abs((fidelity_epochs[epochs_learned - 1] - fidelity_epochs[epochs_learned])
                    /fidelity_epochs[epochs_learned - 1]) < (self.para['converge_accuracy']) / self.update_info['step'])  # 1e-5/2e-1
            if self.update_info['is_converged']:
                if self.update_info['step'] > self.para['step_accuracy']:   # 5e-3
                    self.update_info['step'] /= self.para['step_decay_rate']   # 5
                    print('update step reduces to ' + str(self.update_info['step']))
                    self.update_info['is_converged'] = False

    def print_converge_info(self):
        print(self.para['converge_type'] + ' is converged. Program terminates')
        print('Train ' + str(self.update_info['epochs_learned']) + ' epochs')