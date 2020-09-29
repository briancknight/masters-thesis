import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-3, 0, 100000)
t = np.linspace(1, 4, 4)

fig = plt.figure()

# indicator function: I_0:

plt.axvline(0.0, ymin = 0.25, ymax = 1, color='k', linestyle='--')
# plt.axhline(y=0.0, xmin = 0, color='k', linestyle='--')
plt.plot(x, np.zeros(len(x)), 'k--')
for i in t:
	plt.plot(x, -1/i * np.log(-x), 'b--')

# fill in last approximation

plt.plot(x, -1/t[-1] * np.log(-x), 'b')

plt.ylim(-1, 3)

plt.show()