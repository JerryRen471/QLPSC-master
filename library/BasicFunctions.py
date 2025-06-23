import torch as tc


def sort_dict(a):
    b = dict()
    dict_index = sorted(a.keys())
    for index in dict_index:
        b[index] = a[index]
    return b