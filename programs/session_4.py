# import torch 

# # First tensor
# X = torch.tensor([[1.0, 4.0, 7.0],[2.0, 3.0, 6.0]])

# print(X.dtype)

# print(X[:,1])

# print(10 * (X + 1.0))

# # print(X.max(dim=0))

# print(X @ X.T)

# import numpy as np

# # X.numpy()

# functions = lambda mod : set(f for f in dir(mod) if callable(getattr(mod ,f)))

# #print(", ".join(sorted(functions(torch) & functions(np))))


# # Hardware acceleration --------------------------------------------------

# if torch.cuda.is_available():
#     device = 'cuda'
# elif torch.backends.mps.is_available():
#     device = 'mps'
# else:
#     device = 'cpu'

# #print(device)


# M = torch.tensor([[4,2,1],[4,5,6]])
# M = M.to(device)

# M = torch.tensor([[4,2,1], [4,5,6]], device=device)


# import timeit

# test_1= '''
# import torch
# M = torch.rand((20000,20000))
# M @ M.T
# '''

# test_2 = '''
# import torch
# M = torch.rand((20000,20000), device='mps:0')
# M @ M.T
# '''


# #print(timeit.timeit(stmt=test_1, number=5))
# #print(timeit.timeit(stmt=test_2, number=5))


# #print('-- CPU: %s seconds--' % (time.time() - start_time))

# # M = torch.rand((1000,1000), device='mps:0')
# # #start_time = time.time()
# # timeit.timeit('M @ M.T')
# # #print('-- GPU: %s seconds--' % (time.time() - start_time))

# # ------------------------------------------------------------------------------

# # x = torch.tensor(5.0, requires_grad=True)

# # f = x ** 2

# # f.backward()

# # # print(x.grad)

# # learning_rate = 0.1
# # with torch.no_grad():
# #     x -= learning_rate + x.grad
# #     print(x)


# # x_detached = x.detach()
# # x_detached -= learning_rate * x.grad

# # x.grad.zero_()


# # learning_rate = 0.1
# # x =torch.tensor(5.0, requires_grad=True)
# # for iteration in range(100):
# #     f = x ** 2 # Forward pass
# #     f.backward() # Backward pass
# #     with torch.no_grad():
# #         x -= learning_rate * x.grad # Gradient descent

# #     x.grad.zero_()

# # t = torch.tensor(2.0, requires_grad=True)

# # z = t.sqrt()

# # z += 1

# # #z = z + 1

# # z.backward()

# # ------------------------------------------------------------------------------

# # Linear regression using pytoch 

# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split

# housing = fetch_california_housing()
# X_train_full, X_test, y_train_full, y_test = train_test_split(
#     housing.data, housing.target)

# X_train, X_valid, y_train, y_valid = train_test_split(
#     X_train_full, y_train_full)

# # ------------------------------------------------------------------------------

# # Convert data to pytorch

# X_train = torch.FloatTensor(X_train)
# X_valid = torch.FloatTensor(X_valid)
# X_test  = torch.FloatTensor(X_test)

# means  = X_train.mean(dim=0, keepdim=True)
# stds   = X_train.std(dim=0,  keepdim=True)

# X_train = (X_train - means) / stds
# X_valid = (X_valid - means) / stds
# X_test  = (X_test - means) / stds

# y_train = torch.FloatTensor(y_train).view(-1, 1)
# y_valid = torch.FloatTensor(y_valid).view(-1, 1)
# y_test = torch.FloatTensor(y_test).view(-1, 1)

# torch.manual_seed(42)
# n_features = X_train.shape[-1]
# w = torch.randn((n_features, 1), requires_grad=True)
# b = torch.tensor(0.0,requires_grad=True)

# # ------------------------------------------------------------------------------

# learning_rate = 0.4  # Hyperparameter
# # n_epochs = 20
# # for epoch in range(n_epochs):

# #     y_pred = X_train @ w + b
# #     loss   = ((y_pred - y_train) ** 2).mean()
# #     loss.backward()

# #     # Working with gradient descent
# #     with torch.no_grad():

# #         b -= learning_rate * b.grad
# #         w -= learning_rate * w.grad
# #         b.grad.zero_()
# #         w.grad.zero_()

# #     print(f'Epoch {epoch + 1} / {n_epochs}, Loss: {loss.item()}')


# # X_new = X_test[:3]

# # with torch.no_grad():
# #     y_pred = X_new @ w + b

# # print(b)

# # ------------------------------------------------------------------------------

# import torch.nn as nn

# model = nn.Linear(in_features=n_features, out_features=1)

# # print(model.bias)

# # print(model.weight)

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# mse = nn.MSELoss()

# def train_bgd(model, optimizer, criterion, X_train, y_train, n_epochs):

#     for epoch in range(n_epochs):
#         y_pred = model(X_train)
#         loss = criterion(y_pred, y_train)
#         loss.backward()

#         optimizer.step()
#         optimizer.zero_grad()
#         print(f'Epoch {epoch + 1} / {n_epochs}, Loss: {loss.item()}')


# # train_bgd(model, optimizer, mse, X_train, y_train, n_epochs=20)        


# model = nn.Sequential(
#     nn.Linear(n_features, 50),
#     nn.ReLU(),
#     nn.Linear(50, 40),
#     nn.ReLU(),
#     nn.Linear(40,1)
#     )

# learning_rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# mse = nn.MSELoss()

# # train_bgd(model, optimizer, mse, X_train, y_train, n_epochs=100)   


# from torch.utils.data import TensorDataset, DataLoader

# train_dataset = TensorDataset(X_train, y_train)
# train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

# model = nn.Sequential(
#     nn.Linear(n_features, 50),
#     nn.ReLU(),
#     nn.Linear(50,40),
#     nn.ReLU(),
#     nn.Linear(40, 1)
# )


# device = "cpu" # apple

# model = model.to(device)

# learning_rate = 0.02
# optimizer     = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
# mse = nn.MSELoss()



# # print(torch.device(0))

# # cuda (nvidia) device = "cuda"

# # device = "cpu"

# def train (model, optimizer, criterion, train_loader, n_epochs):
#     model.train()
#     for epoch in range(n_epochs):
#         total_loss = 0.
#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             y_pred = model(X_batch)
#             loss = criterion(y_pred, y_batch)
#             total_loss += loss.item()
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()

#         mean_loss = total_loss / len(train_loader)
#         print(f'Epoch {epoch + 1} / {n_epochs}, Loss: {mean_loss:.4f}')

# train(model, optimizer, mse, train_loader, 20)
        
# def evaluate(model, data_loader, metric_fn, aggregate_fn = torch.mean):
#     model.eval()
#     metrics = []

#     with torch.no_grad():

#         for X_batch, y_batch in data_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             y_pred = model(X_batch)
#             metric = metric_fn(y_pred, y_batch)
#             metrics.append(metric)

#     return aggregate_fn(torch.stack(metrics))


# valid_dataset = TensorDataset(X_valid, y_valid)
# valid_loader  = DataLoader(valid_dataset, batch_size=32)
# valid_mse     = evaluate(model, valid_loader, mse)

# # print(valid_mse)

# import torchmetrics

# def evaluate_tm(model, data_loader, metric):
#     model.eval()
#     metric.reset()

#     with torch.no_grad():
#         for X_batch, y_batch in data_loader:
#             y_pred = model(X_batch)
#             metric.update(y_pred, y_batch)

#     return metric.compute()


# rmse = torchmetrics.MeanSquaredError(squared=False)

# print(evaluate(model, valid_loader, rmse))


# ------------------------------------------------------------------------------

import torch.nn as nn
import torch
import torchmetrics
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

X_train = torch.FloatTensor(X_train)
X_valid = torch.FloatTensor(X_valid)
X_test  = torch.FloatTensor(X_test)

means  = X_train.mean(dim=0, keepdim=True)
stds   = X_train.std(dim=0,  keepdim=True)

X_train = (X_train - means) / stds
X_valid = (X_valid - means) / stds
X_test  = (X_test - means) / stds

y_train = torch.FloatTensor(y_train).view(-1, 1)
y_valid = torch.FloatTensor(y_valid).view(-1, 1)
y_test = torch.FloatTensor(y_test).view(-1, 1)

train_dataset = TensorDataset(X_train, y_train)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = TensorDataset(X_valid, y_valid)
valid_loader  = DataLoader(valid_dataset, batch_size=32)

# ------------------------------------------------------------------------------

class WideAndDeep(nn.Module):

    def __init__(self, n_features):

        super().__init__()

        self.deep_stack = nn.Sequential(
            nn.Linear(n_features, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(30 + n_features, 1)


    def forward(self, X):

        deep_output   = self.deep_stack(X)
        wide_and_deep = torch.concat([X, deep_output], dim=1)

        return self.output_layer(wide_and_deep)

 
def evaluate_tm(model, data_loader, metric): 
    model.eval() 
    metric.reset()  # reset the metric at the beginning 
    with torch.no_grad(): 
        for X_batch, y_batch in data_loader: 
            y_pred = model(X_batch) 
            metric.update(y_pred, y_batch)  # update it at each iteration 
    return metric.compute()  # compute the final result at the end

def train2(model, optimizer, criterion, metric, train_loader, valid_loader, n_epochs):

    history = {"train_losses" : [], "train_metrics": [], "valid metrics": []}

    for epoch in range(n_epochs):

        total_loss = 0.
        metric.reset()

        for X_batch, y_batch in train_loader:
            model.train()
            y_pred = model(X_batch)
            loss  = criterion(y_pred, y_batch)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)

        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history['valid metrics'].append(
            evaluate_tm(model, valid_loader, metric).item())

        print(f'Epoch => {epoch + 1}/{epoch}')
    

model = WideAndDeep(n_features=8)
learning_rate = 0.002
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
mse       = nn.MSELoss()
rmse      = torchmetrics.MeanSquaredError(squared=False)
history   = train2(model, optimizer, mse, rmse, train_loader, valid_loader,n_epochs=20)
