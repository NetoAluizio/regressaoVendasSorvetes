#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[32]:


temperatura = np.array([30, 25, 36, 18, 25, 29, 30, 33, 37, 31, 26, 37, 29, 26, 30, 31, 34, 38])
venda_sorvete = np.array([20, 12, 50, 10, 18, 25, 26, 32, 48, 22, 16, 52, 24, 20, 28, 29, 35, 40])


# In[33]:


df = pd.DataFrame({'temperatura': temperatura, 'venda_sorvete': venda_sorvete})


# In[34]:


df.head()


# In[35]:


plt.plot(df['temperatura'], df['venda_sorvete'], '.')
plt.xlabel('temperatura')
plt.ylabel('sorvetes')
plt.show()


# In[36]:


# Separando as variáveis dependentes e independentes
x = df['temperatura'].to_numpy()
y = df['venda_sorvete'].to_numpy()

# 0.2 equivale a 20% dos resultados para teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2)


# In[37]:


# treinar o modelo
# definir modelo de regressão
modelo = LinearRegression()

# treinar modelo com os dados do treino
modelo.fit(x_treino.reshape(-1, 1), y_treino.reshape(-1, 1))


# In[38]:


# previsão de número de sorvetes a serem vendidos
# y previsto contem a pevisão dos preços do nosso modelo
y_previsto = modelo.predict(x_teste.reshape(-1, 1))


# In[39]:


plt.plot(range(y_previsto.shape[0]), y_previsto, 'r--')
plt.plot(range(y_teste.shape[0]), y_teste, 'b--')
plt.legend(['Sorvetes previstos', 'sorvetes vendidos'])
plt.xlabel('Índice')
plt.ylabel('Sorvetes')
plt.show()


# In[ ]:




