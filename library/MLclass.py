import numpy as np
import torch as tc
from library import Parameters
import math
import copy
import time


class MachineLearning:
    """
    A class for handling machine learning tasks, including dataset loading, preprocessing, feature mapping, and convergence checking.

    Attributes:
        para (dict): Parameters for the machine learning process, deep-copied from input or default.
        input_data (dict): Stores raw and processed input data.
        data_info (dict): Metadata about the dataset, such as number of samples and features.
        tensor_input (tuple): Placeholder for tensor-formatted input data.
        update_info (dict): Tracks training progress, convergence, and update steps.
        tmp (dict): Temporary storage for timing and other intermediate values.
    """

    def __init__(self, para: dict = Parameters.ml()) -> None:
        """
        Initialize the MachineLearning class with parameters and empty data structures.

        Args:
            para (dict, optional): Dictionary of parameters for machine learning. Defaults to Parameters.ml().
        """
        self.para: dict = copy.deepcopy(para)
        self.input_data: dict = dict()
        self.data_info: dict = dict()
        self.tensor_input: tuple = tuple()
        self.update_info: dict = dict()
        self.tmp: dict = dict()

    def initialize_dataset(self) -> None:
        """
        Load and arrange the dataset, then update data_info with the shape of the sorted training data.
        """
        self.load_dataset()
        self.arrange_dataset()
        self.data_info['n_training'], self.data_info['n_feature'] = self.input_data['sorted_train'].shape
        print('Size of the original dataset is %s' % str(self.input_data['sorted_train'].shape))

    def load_dataset(self) -> None:
        """
        Load the dataset from a file specified in parameters. Only supports 'LPS_tomography' dataset type.
        Updates self.input_data['train'] with the loaded data as a torch tensor.
        """
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

    def arrange_dataset(self) -> None:
        """
        Arrange the dataset according to the sorting module specified in parameters.
        Currently only supports random sorting ('rand').
        """
        if self.para['sort_module'] == 'rand':
            self.input_data['sorted_train'] = self.rand_sort_data(self.input_data['train'])

    def rand_sort_data(self, input_data: tc.Tensor) -> tc.Tensor:
        """
        Randomly shuffle the rows of the input data tensor using a fixed seed from parameters.

        Args:
            input_data (tc.Tensor): The input data tensor to shuffle.

        Returns:
            tc.Tensor: The shuffled data tensor.
        """
        np.random.seed(self.para['rand_index_seed'])
        rand_index = np.random.permutation(input_data.shape[0])
        input_data_rand_sorted = input_data[rand_index, :]
        return input_data_rand_sorted

    def feature_map(self, data_mapping) -> tc.Tensor | bool:
        """
        Map input data to a feature space according to the mapping module in parameters.
        For 'tomography', uses a predefined mapping table for quantum state tomography.

        Args:
            data_mapping: The input data to be mapped (array-like or tensor).

        Returns:
            tc.Tensor | bool: The mapped data as a tensor, or False if mapping is not supported.
        """
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

    def generate_update_info(self) -> None:
        """
        Initialize or reset the update_info dictionary with default values for training progress and convergence.
        """
        self.update_info['is_converged'] = 'untrained'
        self.update_info['update_position'] = 'unknown'
        self.update_info['update_direction'] = +1
        self.update_info['step'] = self.para['update_step']
        self.update_info['batch'] = self.para['batch']
        self.update_info['epochs_learned'] = 0
        self.update_info['cost_function_epochs'] = list()
        self.update_info['fidelity_yang_epochs'] = list()
        self.update_info['fidelity_epochs'] = list()

    def calculate_running_time(self, mode: str = 'end') -> None:
        """
        Record the start or end time of an operation for timing purposes.

        Args:
            mode (str, optional): 'start' to record start time, 'end' to record end time. Defaults to 'end'.
        """
        if mode == 'start':
            self.tmp['start_time'] = time.time()
        elif mode == 'end':
            self.tmp['end_time'] = time.time()

    def print_running_time(self) -> None:
        """
        Print the elapsed time between the last recorded start and end times.
        """
        print('This epoch consumes ' + str(self.tmp['end_time'] - self.tmp['start_time']) + ' seconds.')

    def is_converge(self) -> None:
        """
        Check if the training process has converged based on the convergence type in parameters.
        Updates the update_info dictionary accordingly and adjusts the update step if needed.
        """
        epochs_learned: int = self.update_info['epochs_learned']
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

    def print_converge_info(self) -> None:
        """
        Print a message indicating that convergence has been reached and display the number of epochs trained.
        """
        print(self.para['converge_type'] + ' is converged. Program terminates')
        print('Train ' + str(self.update_info['epochs_learned']) + ' epochs')