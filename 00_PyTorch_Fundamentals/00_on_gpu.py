import torch



tensor = torch.tensor([1,2,3])
print(tensor, tensor.device)

# Move tensor to GPU (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor_on_gpu = tensor.to(device)

print(tensor_on_gpu)

# Move tensor back to CPU (numpy)
tensor_on_cpu = tensor_on_gpu.cpu()
print(tensor_on_cpu)
