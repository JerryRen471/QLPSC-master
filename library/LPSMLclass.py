import numpy as np
import torch as tc
from library import Parameters
from library import LPSclass
from library import MLclass
from library import Programclass
import copy
import pickle
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, TensorDataset


class GTN(LPSclass.LPS, MLclass.MachineLearning, Programclass.Program):
    """
    GTN (Generalized Tensor Network) class for quantum state tomography and machine learning.
    Inherits from LPS (tensor network), MachineLearning, and Program classes.

    This class integrates tensor network initialization, dataset handling, training, and evaluation for quantum state learning tasks.

    Attributes:
        environment_left (tuple | list): Left environment tensors for contraction during training.
        environment_right (tuple | list): Right environment tensors for contraction during training.
        lps0 (list): Target state tensors loaded from file.
    """
    def __init__(self, para: dict = Parameters.gtn()) -> None:
        """
        Initialize the GTN class with parameters, dataset, tensor network, and environments.

        Args:
            para (dict, optional): Dictionary of parameters for GTN. Defaults to Parameters.gtn().
        """
        # Initialize Parameters
        LPSclass.LPS.__init__(self)
        MLclass.MachineLearning.__init__(self, para)
        Programclass.Program.__init__(self, dtype=self.para['dtype'], device=self.para['device'])
        print('\n------Training with Nm=%d, Ns=%d------' % (self.para['measure_train'], self.para['sample_num']))
        if len(self.tensor_data) == 0:
            self.initialize_dataset()  # Prepare dataset
            self.input_data['mapped_train'] = self.feature_map(self.input_data['sorted_train'])
            self.input_data['train_dataset'] = TensorDataset(self.input_data['mapped_train'])
            self.input_data['train_loader'] = DataLoader(self.input_data['train_dataset'], batch_size=self.para['batch'], shuffle=True)
            self.generate_tensor_info()  # Initialize tensor info
            self.generate_update_info()
            self.load_target_state()  # Load target state
            self.initialize_lps_gtn()  # Initialize LPS model
        # Environment Preparation
        self.environment_left: tuple | list = tuple()
        self.environment_right: tuple | list = tuple()
        self.lps0: list | None = None

    def generate_tensor_info(self) -> None:
        """
        Initialize tensor_info dictionary with regularization center, length, and bond dimensions.
        """
        self.tensor_info['regular_center'] = 'unknown'
        self.tensor_info['n_length'] = self.data_info['n_feature']
        self.tensor_info['physical_bond'] = self.para['physical_bond']
        self.tensor_info['virtual_bond'] = self.para['virtual_bond']

    def load_target_state(self) -> None:
        """
        Load the target state tensors from a file specified in parameters and convert them to torch tensors.
        Updates self.lps0 with the loaded tensors.
        """
        f = open(self.para['load_state_path'], 'rb')
        self.lps0 = pickle.load(f)
        f.close()
        for k in range(len(self.lps0)):
            self.lps0[k] = tc.from_numpy(self.lps0[k]).to(self.dtype).to(self.device)

    def initialize_lps_gtn(self) -> None:
        """
        Initialize the LPS tensor network for GTN according to the initialization type in parameters.
        Sets up the tensor_data list and regularizes the network.
        """
        if self.para['tensor_initialize_type'] == 'rand':
            tc.manual_seed(self.para['lps_rand_seed'])
            for ii in range(self.tensor_info['n_length']):
                self.tensor_data.append(-2*tc.rand(
                    self.para['purification_bond2'],
                    self.para['virtual_bond'],
                    self.tensor_info['physical_bond'],
                    self.para['virtual_bond'],
                    device=self.device, dtype=self.dtype)+1+1j)
            ii = 0
            self.tensor_data[ii] = -2*tc.rand(
                    self.para['purification_bond2'],
                    1,
                    self.tensor_info['physical_bond'],
                    self.para['virtual_bond'],
                    device=self.device, dtype=self.dtype)+1+1j
            ii = -1
            self.tensor_data[ii] = -2*tc.rand(
                    self.para['purification_bond2'],
                    self.para['virtual_bond'],
                    self.tensor_info['physical_bond'],
                    1,
                    device=self.device, dtype=self.dtype)+1+1j
        # Regularization
        self.lps_regularization(-1)
        self.lps_regularization(0)
        self.tensor_data[0] = self.tensor_data[0] / tc.norm(self.tensor_data[0])
        self.update_info['update_position'] = self.tensor_info['regular_center']

    def start_learning(self, learning_epochs: int = 100) -> None:
        """
        Start the training process for a specified number of epochs.
        Handles batch training, cost function and fidelity calculation, convergence checking, and saving results.

        Args:
            learning_epochs (int, optional): Number of epochs to train. Defaults to 100.
        """
        self.calculate_program_info_time(mode='start')
        self.print_program_info(mode='start')
        if self.update_info['epochs_learned'] == 0:
            self.update_info['cost_function_batchs'] = list()
            self.update_info['fidelity_yang_batchs'] = list()
            self.update_info['fidelity_batchs'] = list()
            for batch_idx, (self.input_data['samples']) in enumerate(self.input_data['train_loader']):
                self.tensor_input = self.input_data['samples'][0]
                self.update_info['batch'] = self.tensor_input.shape[0]
                self.initialize_environment()  # environment
                self.calculate_cost_function()
                self.update_info['cost_function_batchs'].append(self.update_info['cost_function_batch'])
                self.calculate_fidelity_yang(self.tensor_data, self.lps0)
                self.update_info['fidelity_yang_batchs'].append(self.update_info['fidelity_yang_batch'])
                self.calculate_fidelity(self.tensor_data, self.lps0)
                self.update_info['fidelity_batchs'].append(self.update_info['fidelity_batch'])
            self.update_info['cost_function'] = sum(self.update_info['cost_function_batchs']) / len(self.update_info['cost_function_batchs'])
            self.update_info['cost_function_epochs'].append(self.update_info['cost_function'])
            print('\nInitializing ... cost function = ' + str(self.update_info['cost_function']))
            self.update_info['fidelity_yang'] = sum(self.update_info['fidelity_yang_batchs']) / len(self.update_info['fidelity_yang_batchs'])
            self.update_info['fidelity_yang_epochs'].append(self.update_info['fidelity_yang'])
            print('Initializing ... fidelity_yang = ' + str(self.update_info['fidelity_yang']))
            self.update_info['fidelity'] = sum(self.update_info['fidelity_batchs']) / len(self.update_info['fidelity_batchs'])
            self.update_info['fidelity_epochs'].append(self.update_info['fidelity'])
            print('Initializing ... fidelity = ' + str(self.update_info['fidelity']))
        if self.update_info['is_converged'] == 'untrained':
            self.update_info['is_converged'] = False
        print('\nStart to learn to ' + str(learning_epochs) + ' epochs')
        while (self.update_info['epochs_learned'] < learning_epochs) and not (self.update_info['is_converged']):
            print('\n-------- Epoch: %d --------' % (self.update_info['epochs_learned']+1))
            # self.calculate_running_time('start')
            self.update_info['cost_function_batchs'] = list()
            self.update_info['fidelity_batchs'] = list()
            for batch_idx, (self.input_data['samples']) in enumerate(self.input_data['train_loader']):
                self.tensor_input = self.input_data['samples'][0]
                self.update_info['batch'] = self.tensor_input.shape[0]
                self.initialize_environment()  # environment
                self.update_one_loop()
            # self.calculate_running_time(mode='end')
            # self.print_running_time()
            self.update_info['cost_function'] = sum(self.update_info['cost_function_batchs']) / len(self.update_info['cost_function_batchs'])
            self.update_info['cost_function_epochs'].append(self.update_info['cost_function'])
            self.update_info['fidelity_yang'] = sum(self.update_info['fidelity_yang_batchs']) / len(self.update_info['fidelity_yang_batchs'])
            self.update_info['fidelity_yang_epochs'].append(self.update_info['fidelity_yang'])
            self.update_info['fidelity'] = sum(self.update_info['fidelity_batchs']) / len(self.update_info['fidelity_batchs'])
            self.update_info['fidelity_epochs'].append(self.update_info['fidelity'])
            print('cost function = ' + str(self.update_info['cost_function']) + ' at ' + str(self.update_info['epochs_learned'] + 1) + ' epochs.')
            print('fidelity_yang = ' + str(self.update_info['fidelity_yang']) + ' at ' + str(self.update_info['epochs_learned'] + 1) + ' epochs.')
            print('fidelity = ' + str(self.update_info['fidelity']) + ' at ' + str(self.update_info['epochs_learned'] + 1) + ' epochs.')
            self.update_info['epochs_learned'] += 1
            self.is_converge()
        if self.update_info['is_converged']:
            self.print_converge_info()
        else:
            print('Training end, ' + self.para['converge_type'] + ' do not converge.')
        self.save_data()
        self.calculate_program_info_time(mode='end')
        self.print_program_info(mode='end')

    def initialize_environment(self) -> None:
        """
        Initialize the left and right environment tensors for the tensor network contraction during training.
        """
        self.environment_left = list(range(self.tensor_info['n_length'] - 1))
        self.environment_right = list(range(self.tensor_info['n_length'] - 1))
        self.tmp['environment_tmp'] = tc.ones(self.update_info['batch'], device=self.device, dtype=self.dtype)
        self.tmp['environment_tmp'].resize_(self.tmp['environment_tmp'].shape + (1,))
        for ii in range(0, self.tensor_info['n_length']-1):  # 0-8
            self.calculate_environment_next(ii)
        for ii in range(self.tensor_info['n_length'] - 1, 0, -1):
            self.calculate_environment_forward(ii-1)  # 8-0

    def calculate_environment_next(self, environment_index: int) -> None:
        """
        Calculate the left environment tensor at a given index for the tensor network contraction.

        Args:
            environment_index (int): The index at which to calculate the left environment.
        """
        if environment_index == 0:
            environment_left_tmp = tc.einsum('nb,abcd,nc->nad', self.tmp['environment_tmp'], self.tensor_data[environment_index],
                                             self.tensor_input[:, environment_index, :].conj())
            self.environment_left[environment_index] = tc.einsum('naj,nak->njk', environment_left_tmp, environment_left_tmp.conj())
        else:
            tmp = tc.einsum('nbe,abcd,nc->nead',
                            self.environment_left[environment_index - 1],
                            self.tensor_data[environment_index],
                            self.tensor_input[:, environment_index, :].conj())
            self.environment_left[environment_index] = tc.einsum('nead,aefg,nf->ndg',
                                                                 tmp,
                                                                 self.tensor_data[environment_index].conj(),
                                                                 self.tensor_input[:, environment_index, :])

    def calculate_environment_forward(self, environment_index: int) -> None:
        """
        Calculate the right environment tensor at a given index for the tensor network contraction.

        Args:
            environment_index (int): The index at which to calculate the right environment.
        """
        if environment_index == self.tensor_info['n_length'] - 2:
            environment_right_tmp = tc.einsum('nc,abcd,nd->nab', self.tensor_input[:, environment_index+1, :].conj(),
                                              self.tensor_data[environment_index+1], self.tmp['environment_tmp'])
            self.environment_right[environment_index] = tc.einsum('naj,nak->njk', environment_right_tmp, environment_right_tmp.conj())
        else:
            tmp = tc.einsum('nc,abcd,ndg->nabg',
                            self.tensor_input[:, environment_index+1, :].conj(),
                            self.tensor_data[environment_index+1],
                            self.environment_right[environment_index+1])
            self.environment_right[environment_index] = tc.einsum('nabg,aefg,nf->nbe',
                                                                  tmp,
                                                                  self.tensor_data[environment_index+1].conj(),
                                                                  self.tensor_input[:, environment_index+1, :])

    def calculate_cost_function(self) -> None:
        """
        Calculate the negative log-likelihood cost function for the current batch and update update_info.
        """
        if self.update_info['update_position'] != 0:
            print('go check your code')
        tmp_tensor0 = tc.einsum('nc,acd->nad', self.tensor_input[:, 0, :].conj(), self.tensor_data[0][:, 0, :, :])
        tmp_tensor = tc.einsum('naj,nak->njk', tmp_tensor0, tmp_tensor0.conj())
        tmp_inner_product = tc.einsum('nab,nab->n', tmp_tensor, self.environment_right[0]).cpu().numpy()  # P(v)
        tmp_Z = tc.einsum('abcd,abcd->', self.tensor_data[0], self.tensor_data[0].conj())
        self.update_info['cost_function_batch'] = (np.log(tmp_Z.cpu().numpy()) -
                                                   sum(np.log(tmp_inner_product)) / self.update_info['batch']).real

    def update_one_loop(self) -> None:
        """
        Perform a single left-to-right and right-to-left update sweep over the tensor network for one training batch.
        Updates the tensors, environments, and cost/fidelity metrics.
        """
        if self.tensor_info['regular_center'] != 0:
            self.lps_regularization(-1)
            self.lps_regularization(0)
        self.update_info['update_position'] = self.tensor_info['regular_center']
        self.update_info['update_direction'] = +1
        while self.tensor_info['regular_center'] < self.tensor_info['n_length'] - 1:
            self.update_lps_once()
            self.lps_regularization(self.update_info['update_position'] + self.update_info['update_direction'])
            self.update_info['update_position'] = self.tensor_info['regular_center']
            self.calculate_environment_next(self.update_info['update_position']-1)
        self.update_info['update_direction'] = -1
        while self.tensor_info['regular_center'] > 0:
            self.update_lps_once()
            self.lps_regularization(self.update_info['update_position'] + self.update_info['update_direction'])
            self.update_info['update_position'] = self.tensor_info['regular_center']
            self.calculate_environment_forward(self.update_info['update_position'])
        self.tensor_data[0] /= (self.tensor_data[0]).norm()
        self.calculate_cost_function()
        self.update_info['cost_function_batchs'].append(self.update_info['cost_function_batch'])
        self.calculate_fidelity_yang(self.tensor_data, self.lps0)
        self.update_info['fidelity_yang_batchs'].append(self.update_info['fidelity_yang_batch'])
        self.calculate_fidelity(self.tensor_data, self.lps0)
        self.update_info['fidelity_batchs'].append(self.update_info['fidelity_batch'])

    def update_lps_once(self) -> None:
        """
        Compute the gradient and update the current tensor in the network at the regularization center.
        Updates the tensor_data and gradient information.
        """
        # Calculate gradient
        tmp_index1 = self.tensor_info['regular_center']
        tmp_tensor_current = self.tensor_data[tmp_index1]
        if tmp_index1 == 0:
            tmp = tc.einsum('acd,nc->nad', self.tensor_data[0][:, 0, :, :], self.tensor_input[:, 0, :].conj())
            tmp_tensor1 = tc.einsum('nam,nc,nmk->nack', tmp, self.tensor_input[:, 0, :],
                                    self.environment_right[0]).reshape(self.update_info['batch'], -1)  # P'(v)
            tmp_inner_product = tc.einsum('nam,nak,nmk->n', tmp, tmp.conj(), self.environment_right[0]).view(-1,1).t()  # P(v)
        elif tmp_index1 == (self.tensor_info['n_length']-1):
            tmp = tc.einsum('abc,nc->nab', self.tensor_data[tmp_index1][:, :, :, 0], self.tensor_input[:, tmp_index1, :].conj())
            tmp_tensor1 = tc.einsum('nam,nmk,nc->nakc', tmp, self.environment_left[tmp_index1-1],
                                    self.tensor_input[:, tmp_index1, :]).reshape(self.update_info['batch'], -1)  # P'(v)
            tmp_inner_product = tc.einsum('nmk,nam,nak->n', self.environment_left[tmp_index1-1], tmp, tmp.conj()).view(-1, 1).t()  # P(v)
        else:
            tmp = tc.einsum('abcd,nc->nabd', self.tensor_data[tmp_index1], self.tensor_input[:, tmp_index1, :].conj())
            tmp_tensor1 = tc.einsum('naim,nij,nc,nmk->najck', tmp, self.environment_left[tmp_index1-1], self.tensor_input[:, tmp_index1, :],
                                    self.environment_right[tmp_index1]).reshape(self.update_info['batch'], -1)  # P'(v)
            tmp_inner_product = tc.einsum('nij,naim,najk,nmk->n', self.environment_left[tmp_index1-1], tmp, tmp.conj(), self.environment_right[tmp_index1]).view(-1,1).t()  # P(v)
        tmp_Z = tc.einsum('abcd,abcd->', self.tensor_data[tmp_index1], self.tensor_data[tmp_index1].conj())
        tmp_tensor1 = ((1 / tmp_inner_product).mm(tmp_tensor1)).reshape(tmp_tensor_current.shape)
        self.tmp['gradient'] = tmp_tensor_current/tmp_Z - tmp_tensor1 / self.update_info['batch']
        x = tc.einsum('ijmk,ijmk->',self.tmp['gradient'],tmp_tensor_current.conj())
        print(abs(x))  # Check gradient
        # update LPS
        tmp_tensor_current -= self.update_info['step'] * self.tmp['gradient']
        self.tensor_data[self.update_info['update_position']] = tmp_tensor_current

    def calculate_inner_product_new(self, mps_l0: list, mps_l1: list) -> tc.Tensor:
        """
        Calculate the inner product between two matrix product states (MPS).

        Args:
            mps_l0 (list): First MPS tensor list.
            mps_l1 (list): Second MPS tensor list.

        Returns:
            tc.Tensor: The inner product as a tensor.
        """
        MPS_l0 = copy.deepcopy(mps_l0)
        s0 = MPS_l0[0].shape
        MPS_l0[0] = MPS_l0[0].reshape(s0[0], s0[2], s0[3])
        MPS_l1 = copy.deepcopy(mps_l1)
        s1 = MPS_l1[0].shape
        MPS_l1[0] = MPS_l1[0].reshape(s1[0], s1[2], s1[3])
        tmp0 = MPS_l0[0]
        tmp0 = tc.einsum('abc,ade->bcde', [tmp0, MPS_l0[0].conj()])
        tmp0 = tc.einsum('bcde,fdg->bcefg', [tmp0, MPS_l1[0]])
        tmp0 = tc.einsum('bcefg,fbh->cegh', [tmp0, MPS_l1[0].conj()])
        for i in range(1, len(MPS_l0)):
            tmp0 = tc.einsum('abcd,eafg->bcdefg', [tmp0, MPS_l0[i]])
            tmp0 = tc.einsum('bcdefg,ebhk->cdfghk', [tmp0, MPS_l0[i].conj()])
            tmp0 = tc.einsum('cdfghk,ochm->dfgkom', [tmp0, MPS_l1[i]])
            tmp0 = tc.einsum('dfgkom,odfq->gkmq', [tmp0, MPS_l1[i].conj()])
        inner_product = tc.squeeze(tmp0)
        return inner_product

    def calculate_fidelity_yang(self, mps_l0: list, mps_l1: list) -> None:
        """
        Calculate the fidelity (Yang's definition) between two MPS states and update update_info.

        Args:
            mps_l0 (list): First MPS tensor list.
            mps_l1 (list): Second MPS tensor list.
        """
        MPS_l0 = copy.deepcopy(mps_l0)
        MPS_l1 = copy.deepcopy(mps_l1)
        rho11 = self.calculate_inner_product_new(MPS_l0, MPS_l0)
        rho22 = self.calculate_inner_product_new(MPS_l1, MPS_l1)
        rho12 = self.calculate_inner_product_new(MPS_l0, MPS_l1)
        fide_new = (rho12 / (rho11 * rho22) ** 0.5).cpu()
        self.update_info['fidelity_yang_batch'] = float("%.16f" % abs(fide_new))

    def contract_LPS_new(self, lps_l: list) -> tc.Tensor:
        """
        Contract an LPS (tensor network) into a density matrix. First to MPO, then to density matrix.

        Args:
            lps_l (list): List of LPS tensors.

        Returns:
            tc.Tensor: The contracted density matrix as a tensor.
        """
        LPS_l = copy.deepcopy(lps_l)
        n = len(LPS_l)
        tmp = list(range(n))
        for i in range(n):
            s = LPS_l[i].shape
            tmp[i] = tc.einsum('abcd,aefg->bcdefg', LPS_l[i], LPS_l[i].conj())
            tmp[i] = tmp[i].permute(4, 0, 3, 1, 2, 5)
            tmp[i] = tmp[i].reshape(s[2], s[1] ** 2, s[2], s[3] ** 2)
        tmp1 = tmp[0]
        for i in range(1, n):
            tmp1 = tc.tensordot(tmp1, tmp[i], ([-1], [1]))
        a = list(range(2, 2 * (n + 1), 2))
        b = list(range(3, 2 * (n + 1), 2))
        o = [1] + a + [0] + b
        tmp1 = tmp1.permute(o)  # 改变长度需改变(1,2,4,...,2n,0,3,5,...,2n-1,2n+1)
        s1 = tmp1.shape
        rho = tmp1.reshape(s1[0], 2 ** n, 2 ** n, s1[-1])
        rho = tc.einsum('abca->bc', rho)
        return rho

    def calculate_fidelity(self, lps0: list, lps1: list) -> None:
        """
        Calculate the fidelity between two density matrices contracted from LPS tensors and update update_info.

        Args:
            lps0 (list): First LPS tensor list.
            lps1 (list): Second LPS tensor list.
        """
        lps0_tmp = copy.deepcopy(lps0)
        rho0 = self.contract_LPS_new(lps0_tmp)  # Density matrix from training
        rho0 = rho0.cpu().numpy()
        lps1_tmp = copy.deepcopy(lps1)
        rho1 = self.contract_LPS_new(lps1_tmp)
        rho1 = rho1.cpu().numpy()
        fide = np.trace(sqrtm(np.dot(np.dot(sqrtm(rho1), rho0), sqrtm(rho1))))
        self.update_info['fidelity_batch'] = float("%.16f" % abs(fide))

    def save_data(self) -> None:
        """
        Save the training results (cost function and fidelity metrics) to a file specified in parameters.
        """
        dic = {'NLLloss': self.update_info['cost_function_epochs'], 'fidelity_yang': self.update_info['fidelity_yang_epochs'], 'fidelity': self.update_info['fidelity_epochs']}
        f = open(self.para['save_result_path']+'%d_%d_chi%d.txt' % (
            self.para['measure_train'], self.para['sample_num'], self.para['virtual_bond']), 'wb')
        pickle.dump(dic, f)
        f.close()