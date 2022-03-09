import torch


print(torch.cuda.is_available())

X_train = torch.FloatTensor([0.,1.,2.])
X_train = X_train.to(device)
print(X_train.is_cuda)
