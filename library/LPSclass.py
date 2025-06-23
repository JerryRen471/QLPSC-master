import numpy as np
import torch as tc
from library import TNclass


class LPS(TNclass.TensorNetwork):
    def __init__(self):
        # Initialize Parameters
        TNclass.TensorNetwork.__init__(self)

    def lps_regularization(self, regular_center):
        if regular_center == -1:
            regular_center = self.tensor_info['n_length']-1
        if self.tensor_info['regular_center'] == 'unknown':
            self.tensor_info['regular_center'] = 0
            while self.tensor_info['regular_center'] < self.tensor_info['n_length']-1:
                self.move_regular_center2next()
        while self.tensor_info['regular_center'] < regular_center:
            self.move_regular_center2next()
        while self.tensor_info['regular_center'] > regular_center:
            self.move_regular_center2forward()

    def move_regular_center2next(self):
        tensor_index = self.tensor_info['regular_center']
        type = self.tensor_data[tensor_index].dtype
        device = self.tensor_data[tensor_index].device
        merged_tensor = tc.tensordot(self.tensor_data[tensor_index], self.tensor_data[tensor_index+1], ([3], [1]))
        s1 = merged_tensor.shape
        merged_tensor = merged_tensor.reshape(s1[0] * s1[1] * s1[2], s1[3] * s1[4] * s1[5])
        merged_tensor = merged_tensor.cpu().numpy()
        u, lm, v = np.linalg.svd(merged_tensor, full_matrices=False)
        u, lm, v = tc.from_numpy(u), tc.from_numpy(lm), tc.from_numpy(v)
        u, lm, v = u.to(dtype=type, device=device), lm.to(dtype=type, device=device), v.to(dtype=type, device=device)
        bdm = min(self.tensor_info['virtual_bond'], len(lm))
        u = u[:, :bdm]
        lm = tc.diag(lm[:bdm])
        v = v[:bdm, :]
        self.tensor_data[tensor_index] = u.reshape(s1[0], s1[1], s1[2], bdm)
        v = tc.mm(lm, v)
        s2 = self.tensor_data[tensor_index+1].shape
        self.tensor_data[tensor_index + 1] = v.reshape(v.shape[0], s2[0], s2[2], s2[3])
        self.tensor_data[tensor_index + 1] = tc.transpose(self.tensor_data[tensor_index+1], 0, 1)
        self.tensor_info['regular_center'] += 1

    def move_regular_center2forward(self):
        tensor_index = self.tensor_info['regular_center']
        type = self.tensor_data[tensor_index].dtype
        device = self.tensor_data[tensor_index].device
        merged_tensor = tc.tensordot(self.tensor_data[tensor_index-1], self.tensor_data[tensor_index], ([3], [1]))
        s1 = merged_tensor.shape
        merged_tensor = merged_tensor.reshape(s1[0] * s1[1] * s1[2], s1[3] * s1[4] * s1[5])
        merged_tensor = merged_tensor.cpu().numpy()
        u, lm, v = np.linalg.svd(merged_tensor, full_matrices=False)
        u, lm, v = tc.from_numpy(u), tc.from_numpy(lm), tc.from_numpy(v)
        u, lm, v = u.to(dtype=type, device=device), lm.to(dtype=type, device=device), v.to(dtype=type, device=device)
        bdm = min(self.tensor_info['virtual_bond'], len(lm))
        u = u[:, :bdm]
        lm = tc.diag(lm[:bdm])
        v = v[:bdm, :]
        self.tensor_data[tensor_index] = v.reshape(v.shape[0], s1[3], s1[4], s1[5])
        self.tensor_data[tensor_index] = tc.transpose(self.tensor_data[tensor_index], 0, 1)
        u = tc.mm(u, lm)
        s = self.tensor_data[tensor_index-1].shape
        self.tensor_data[tensor_index-1] = u.reshape(s[0], s[1], s[2], u.shape[1])
        self.tensor_info['regular_center'] -= 1