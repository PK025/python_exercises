#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

from backpropagation import BackpropagationNetwork


x = np.array([0, 0, 0.0])
x = np.c_[x, [1, 0, 0]]
x = np.c_[x, [0, 1, 0]]
x = np.c_[x, [1, 1, 0]]
x = np.c_[x, [0, 0, 1]]
x = np.c_[x, [1, 0, 1]]
x = np.c_[x, [0, 1, 1]]
x = np.c_[x, [1, 1, 1]]
x = x.T

t = np.array([0, 0.0])
t = np.c_[t, [1, 0]]
t = np.c_[t, [1, 0]]
t = np.c_[t, [0, 1]]
t = np.c_[t, [1, 0]]
t = np.c_[t, [0, 1]]
t = np.c_[t, [0, 1]]
t = np.c_[t, [1, 1]]
t = t.T


reps = 1000
E_history = np.zeros(reps)
            
bn = BackpropagationNetwork(3, 2)
bn.set_mi(2)

for n in range(reps):
    E = bn.train(x, t)
    E_history[n] = E
    

pred = bn.predict(x)

print('results(input, predicted, real):')
print(E)
for i in range(8):
    print(x[i], pred[i], t[i])

plt.plot( E_history, 'r-')
plt.grid(True)
plt.title(u'Error over time')
plt.ylabel(u'E  ', rotation=0)
plt.xlabel(u'Iterations')
plt.show()
    