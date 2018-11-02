import numpy as np

x = np.ones((1, 2, 3))

print('x:', x)
print('x.transpose(1, 0, 2)', np.transpose(x, (1, 0, 2)))

print('shape x', x.shape)

print('shape of transposed x', np.transpose(x, (1, 0, 2)).shape)