import numpy as np
from Amber.Core.Player import LocalPlayer
from Amber.NN.OpFacotry import OpFactory
from Amber.NN.Layers import DenseLayer


opf = OpFactory(LocalPlayer())
opf.use_as_default()

dense = DenseLayer(3, 3)

x = opf.new_node(np.array([[1, 2, 3]]))
y = opf.new_node(np.array([[4, 5, 6]]))
z = x * y
print(z.get())

zz = dense(z)
print(zz.get())

zzz = opf.mean(zz, n_heading_axes=2)
print(zzz.get())

grad_on_x, grad_on_y = opf.gradient_on(zzz, [x, y])
print(grad_on_x.get(), grad_on_y.get())
