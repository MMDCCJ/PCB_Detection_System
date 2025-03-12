import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
import torch
# import torchvision
print(torch.__version__)      # 打印 PyTorch 版本
# print(torchvision.__version__) 
print(torch.cuda.is_available())  # 检查是否支持 CUDA
print(torch.version.cuda)    