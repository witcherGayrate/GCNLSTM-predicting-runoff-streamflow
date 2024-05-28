import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
value = np.load('./value_matrix.npy')
runoff_value = value[:,:,-1]
print(f'value.shape :{value.shape}')
print(f'runoff_value.shape :{runoff_value.shape}')

correlation_matrix = np.corrcoef(runoff_value)
print(f'correlation_matrix.shape :{correlation_matrix.shape}')



plt.imshow(correlation_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.show()
