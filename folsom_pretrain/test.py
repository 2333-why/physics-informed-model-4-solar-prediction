import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)

print("npu available:", torch.npu.is_available())
print("device count:", torch.npu.device_count())

torch.npu.set_device("npu:0")

x = torch.randn(2,3).to("npu:0")

print(x)
print(x.device)