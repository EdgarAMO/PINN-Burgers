# PINN-Burgers
Burgers equation solved by PINN in PyTorch

# Libraries you need
  1.- PyTorch
  2.- Numpy
  3.- SciPy
  4.- Matplotlib
  
# Procedure explained:
  1.- Boundary train data collected in a [100, 2] numpy array: pairs of (x, t)
  2.- Boundary solution data collected in a [100, 2] numpy array
  3.- Collocation points come in a [10100, 2] numpy array
  4.- These arrays are passed to the PINN
  5.- These arrays are transformed into torch tensors
  6.- The net is created manualy, although you should better inherit from torcn.nn.Module
  7.- The net takes a two-input vector and returns a one-output vector 
  8.- LGFGS is used to update the net gradientes (this is a full-batch prodecure)
  9.- Xavier initialization didn't work out for me, so I left it alone
  10.- The net represents the u function, f is a function of the net
  11.- f is the differential equation equaled to zero
  12.- Both the boundary conditions and the PDE losses (u_loss and f_loss) are added together
  13.- The sum of these two losses is the minimized using the torch.optimizer
  
# How do I run the code?
  - Just run the module
  - Type pinn.plot() in the console to check the plot
  
# Thank you for downloading!
