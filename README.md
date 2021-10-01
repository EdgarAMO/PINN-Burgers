# PINN-Burgers
Burgers equation solved by PINN in PyTorch

# Libraries you need
  - PyTorch
  - Numpy
  - SciPy
  - Matplotlib
  
# Procedure explained:
  - Boundary train data collected in a [100, 2] numpy array: pairs of (x, t)
  - Boundary solution data collected in a [100, 2] numpy array
  - Collocation points come in a [10100, 2] numpy array: paris of (x, t)
  - These arrays are passed to the PINN
  - These arrays are transformed into torch tensors
  - The net is created manualy, although you should better inherit from torch.nn.Module
  - The net takes a two-input vector and returns a one-output vector 
  - LBFGS is used to update the net gradientes (this is a full-batch prodecure)
  - Xavier initialization didn't work out for me, so I left it alone
  - The net represents the u function, f is a function of the net
  - f is the differential equation equaled to zero
  - Both the boundary conditions and the PDE losses (u_loss and f_loss) are added together
  - The sum of these two losses is the minimized using the torch.optimizer
  
# How do I run the code?
  - Just run the module
  - Type pinn.plot() in the console to check the plot
  
# Thank you for downloading!
