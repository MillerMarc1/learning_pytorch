import torch
import numpy as np

# Creating a tensor

# scalar
scalar = torch.tensor(7)
print(scalar) # tensor(7)

# scalars have no dimensions
print(scalar.ndim)  # 0


# vector
vector = torch.tensor([7,7])
print(vector)      # tensor([7, 7])
print(vector.ndim) # 1


# MATRIX
MATRIX = torch.tensor([[7,8],[9,10]])
print(MATRIX)        # tensor([[ 7,  8], [ 9, 10]])
print(MATRIX.ndim)   # 2
print(MATRIX[0])     # tensor([7, 8])
print(MATRIX.shape)  # torch.Size([2, 2])


# TENSOR
TENSOR = torch.tensor([[[1, 2, 3], [3, 6, 9], [2, 4, 5]]])
print(TENSOR)         # tensor([[[1, 2, 3], [3, 6, 9], [2, 4, 5]]])
print(TENSOR.ndim)    # 3
print(TENSOR.shape)   # torch.Size([1, 3, 3])


print("-"*40)


# Random Tensors

random_tensor = torch.rand(3, 4)
print(random_tensor)

random_image_tensor = torch.rand(3,244,244)
print(random_image_tensor.shape) # torch.Size([3, 244, 244])
print(random_image_tensor.ndim)  # 3

one_to_ten_by_2 = torch.arange(start = 1, end = 11, step = 2)
print(one_to_ten_by_2)



# Tensor Attributes
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # what datatype the tensor is (e.g. float32 or float16
                               device=None, # what device is your tensor on
                               requires_grad=False # whether or not to track gradients
                               )


some_tensor = torch.rand(3, 4)
print(some_tensor)                                    # tensor([[0.4968, 0.8156, 0.4440, 0.1621],
                                                      #         [0.2422, 0.0706, 0.4422, 0.5786],
                                                      #         [0.3119, 0.9549, 0.4922, 0.5603]])

print(f"Datatype of tensor: {some_tensor.dtype}")     # Datatype of tensor: torch.float32
print(f"Shape of tensor: {some_tensor.shape}")        # Shape of tensor: torch.Size([3, 4])
print(f"Device tensor is on: {some_tensor.device}")   # Device tensor is on: cpu



# Manipulating Tensors

tensor = torch.tensor([1,2,3])
tensor = tensor + 100
print(tensor)                   # tensor([101, 102, 103])

tensor = tensor / 2
print(tensor)                   # tensor([50.5000, 51.0000, 51.5000])



# Matrix multiplication
tensor = torch.tensor([1, 2, 3])
print(tensor, "*", tensor)  # tensor([1, 2, 3]) * tensor([1, 2, 3])
print(tensor * tensor)      # tensor([1, 4, 9])


print(torch.matmul(tensor, tensor))  # tensor(14)

print(tensor @ tensor)   # tensor(14)


# Shapes for matrix multiplication
tensor_A = torch.tensor([[1,2],
                         [3,4],
                         [5,6]])

tensor_B = torch.tensor([[7,10],
                        [8,11],
                        [9,12]])

# torch.mm() is the same as torch.matmul()
print(torch.mm(tensor_A, tensor_B.T)) 



# min, max, mean, sum, etc..

x = torch.arange(0, 100, 10)

print(torch.min(x))
print(x.min())

print(torch.max(x))
print(x.max())

# changing type to float32 because mean does not accept long data type
print(torch.mean(x.type(torch.float32)))
print(x.type(torch.float32).mean())


tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}") # 8
print(f"Index where min value occurs: {tensor.argmin()}") # 0


# Indexing

x = torch.arange(1,10).reshape(1,3,3)

print(x)
print(x.shape)

# dim=0
print(x[0]) # [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Index the middle bracket (dim=1)
print(x[0][0])  # [1,2,3]

# Index the most inner bracket (dim=2)
print(x[0][0][0])   # 1

# Get all values of the 0th dimension and 1st dimension but only index 1 of 2nd dim
print(x[:,:,1]) # [2,5,8]


# Pytorch and NumPy
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(tensor)


tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor)


# Reproducibility

RANDOM_SEED = 42
torch.manual_seed(seed=RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(seed=RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C)
print(random_tensor_D)


# Check for GPU
print(torch.cuda.is_available()) # True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)





