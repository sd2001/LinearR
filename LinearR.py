import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X_numpy,y_numpy=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
X=torch.from_numpy(X_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))
y=y.view(y.shape[0],1)
n_samples,n_features=X.shape
input_features=n_features
output_features=1
model=nn.Linear(input_features,output_features)

criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

num_epochs=1000

for epoch in range(num_epochs):
    y_predicted=model(X)
    loss=criterion(y_predicted,y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if(epoch%10)==0:
        print(f'epoch : {epoch}, loss = {loss.item():.4f}')

predicted=model(X).detach().numpy()
plt.plot(X_numpy,y_numpy,'ro')
plt.plot(X_numpy,predicted,'b')
plt.show()

