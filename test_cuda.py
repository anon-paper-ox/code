import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())

print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())

x = torch.randn(2048, 2048, device="cuda")
y = x @ x
print("ok:", float(y[0,0]))