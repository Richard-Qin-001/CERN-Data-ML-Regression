import torch
import torchvision

print("PyTorch 版本:", torch.__version__)
print("torchvision版本:",torchvision.__version__)
print("CUDA 版本:", torch.version.cuda)
print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))
else:
    print("GPU 不可用")

import tensorflow as tf
print(tf.__version__)

import tensorflow as tf
print(f"TensorFlow 版本: {tf.__version__}")
print(f"GPU 是否可用: {tf.config.list_physical_devices('GPU')}")