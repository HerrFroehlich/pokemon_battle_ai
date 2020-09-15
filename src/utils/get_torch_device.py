import torch


GLOBAL_TORCH_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('#- Using global device for torch:', GLOBAL_TORCH_DEVICE)