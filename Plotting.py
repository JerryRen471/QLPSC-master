import torch as tc
import matplotlib.pyplot as plt
import pickle


para = dict()
para['num_f'] = 10
para['chi'] = 16
para['measure_train'] = 50
para['sample_num'] = 1000


f = open('/Users/wenjun/code/Result/Random_Glps/chain%d/normal_1_3_0/%d_%d_chi%d.txt' % (para['num_f'],para['measure_train'],para['sample_num'],para['chi']), 'rb')
dic = pickle.load(f)
f.close()

NLLloss_List = dic['NLLloss']
fidelity_yang = dic['fidelity_yang']
fidelity = dic['fidelity']

'''画图'''
x1 = tc.linspace(0, len(NLLloss_List), len(NLLloss_List))
y1 = NLLloss_List
a1 = min(y1)
d1 = max(y1)

plt.figure()
plt.ylim(a1-0.02*a1, d1+0.02*d1)
plt.scatter(x1.numpy(), y1)
plt.xlabel('Epoch')
plt.ylabel('NLLloss')
plt.title('LPS,L:%d,$\chi$:%d,$N_m:%d$'%(para['num_f'],para['chi'], para['measure_train']))
plt.tight_layout()
plt.show()

# "===============杨保真度==============="
x3 = tc.linspace(0, len(fidelity_yang), len(fidelity_yang))
y3 = tc.zeros([len(fidelity_yang)], dtype=tc.float64)
for i in range(len(fidelity_yang)):
    y3[i] = float(fidelity_yang[i])
w1 = min(y3)
w2 = max(y3)
plt.figure()
# plt.ylim(w1-0.01*w1, w2+0.01*w2)
plt.scatter(x3, y3)
plt.xlabel('Epoch')
plt.ylabel('Fidelity_yang')
plt.title('LPS,L:%d,$\chi$:%d,$N_m$:%d' % (para['num_f'], para['chi'], para['measure_train']))
plt.tight_layout()
plt.show()

# "===============保真度==============="
x8 = tc.linspace(0, len(fidelity), len(fidelity))
y8 = tc.zeros([len(fidelity)], dtype=tc.float64)
# x8 = list(range(len(y8)))
for i in range(len(fidelity)):
    y8[i] = float(fidelity[i])
w1 = min(y8)
w2 = max(y8)
plt.figure()
# plt.ylim(w1-0.01*w1, w2+0.01*w2)
plt.scatter(x8, y8)
plt.xlabel('Epoch')
plt.ylabel('Fidelity')
plt.title('LPS,L:%d,$\chi$:%d,$N_m$:%d' % (para['num_f'], para['chi'], para['measure_train']))
plt.tight_layout()
plt.show()


