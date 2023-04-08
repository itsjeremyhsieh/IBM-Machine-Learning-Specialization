# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
data = {'knn': 0.2055835605152935,
        'Linear Regression': 0.21028961946031516,
        'ElasticNet': 0.2109446644463765,
        'Ridge': 0.21028956245332192,
        'Lasso': 0.2109446644463765,
        'Neural_Network': 0.2366,
        }
#%%
x = data.keys()
y = data.values()
plt.figure(figsize=(20,10))
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 11}

plt.rc('font', **font)
plt.bar(x = x,height=y,color=['lightblue']*7 + ['orange']*2)
plt.ylabel('RMSE')
plt.title('Model RMSE Comparison')
