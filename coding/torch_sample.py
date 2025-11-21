'''
pytorch basic
'''
import torch
import numpy as np

# tensor
def pytorch_tensor():
    # pytorch tensor examples
    ## create an un-initialized 5x3 tensor
    x = torch.empty(5, 3)
    # x: tensor with uninitialized values

    ## create a random tensor
    x = torch.rand(5, 3)
    ### x: tensor with random values

    ## create a tensor filled with zeros and of dtype long
    x = torch.zeros(5, 3, dtype=torch.long)
    ### x: tensor filled with zeros
    ### x.shape: torch.Size([5, 3])

    ## create a tensor directly from data
    x = torch.tensor([5.5, 3])
    print(x) # result: tensor with values [5.5, 3.0]
    print(x.shape) # result: torch.Size([2])

    ## create a unit tensor based on an existing tensor
    x = x.new_ones(5, 3, dtype=torch.double)
    print(x) # result: tensor filled with ones
    print(x.shape) # result: torch.Size([5, 3]) 

    ## create a tensor based on the size of an existing tensor
    ## and re-define as float
    x = torch.randn_like(x, dtype=torch.float)
    print(x) # result: tensor with random values
    print(x.shape) # result: torch.Size([5, 3])

    ## print a tensor's dimension
    print(x.size()) # result: torch.Size([5, 3])

    ## plus two tensors
    y = torch.rand(5, 3)
    print(x + y) # result: element-wise addition of x and y

    ## print first row of tensor
    print(x[0, :]) # result: first row of tensor x

    ## resize a [4, 4] tensor to [2, 8] and [1, 16]
    x = torch.randn(4, 4)
    print(x) # result: tensor with random values
    y = x.view(2, 8)
    print(x.size(), y.size()) # result: reshaped tensor [2, 8]
    z = x.view(1, 16)
    print(x.size(), z.size()) # result: reshaped tensor [1, 16]

    ## get number from a tensor
    x = torch.randn(1)
    print(x) # result: tensor with a single random value
    print(x.item()) # result: the single value as a standard Python number

def pytorch_numpy():
    # pytorch and numpy conversion examples
    ## tensor to numpy array
    a = torch.ones(5)
    print(a) # result: tensor filled with ones
    b = a.numpy()
    print(b) # result: numpy array filled with ones

    ## tensor plus 1
    a.add_(1)
    print(a) # result: tensor filled with twos
    print(b) # result: numpy array filled with twos (shared memory)

    ## create tensor from numpy array
    a = np.ones(5)
    b = torch.from_numpy(a)
    print(a) # result: numpy array filled with ones
    print(b) # result: tensor filled with ones

    ## numpy array plus 1
    np.add(a, 1, out=a)
    print(a) # result: numpy array filled with twos
    print(b) # result: tensor filled with twos (shared memory)

# auto differential
def pytorch_tensor_auto_differential():
    ## create new tensor...
    x = torch.ones(2, 2, requires_grad=True)
    print(x)
    ### x: tensor([[1., 1.], [1., 1.]], requires_grad=True)

    ## play with tensor
    y = x + 2
    ### y: x add 2 each element

    ## play with y
    z = y * y * 3
    out = z.mean()
    ### z: tensor([[27., 27.], [27., 27.]], grad_fn=<MulBackward0>)
    ### out: mean from all elements in tensor, 27

def pytorch_gradient():
    ## fake input
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    z = y * y * 3
    out = z.mean()

    ## backward to tensor
    out.backward()

    ## print gradient, d(out)/dx
    print(x.grad)
    ### x.grad: 0.25 * Σ3(x + 2) ^ 2


if __name__ == "__main__":
    # pytorch_tensor()
    # pytorch_numpy()
    # pytorch_tensor_auto_differential()
    pytorch_gradient()