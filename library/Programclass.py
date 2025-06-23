import torch as tc
import time
from library import BasicFunctions

class Program:

    def __init__(self, dtype=tc.complex128, device='cuda'):
        self.program_info = dict()
        self.dtype = dtype
        self.device = device

    def calculate_program_info_time(self, mode='start'):
        if mode == 'start':
            self.program_info['start_time'] = time.time()
        if mode == 'end':
            self.program_info['end_time'] = time.time()

    def print_program_info(self, mode='start'):
        if mode == 'start':
            print('This program starts at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print(BasicFunctions.sort_dict(self.para))
        elif mode == 'end':
            print('This program ends at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print(
                'This program consumes ' +
                str(self.program_info['end_time'] - self.program_info['start_time']) +
                ' seconds of time.')