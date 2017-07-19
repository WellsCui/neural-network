from cs231n.classifiers.convnet import *
from cs231n.solver import *
from cs231n.data_utils import *


data = get_CIFAR10_data(cifar10_dir = '/input')
for k in data:
    print('%s: ' % k, data[k].shape)

model = ResNet()

solver = Solver(model, data,
                num_epochs=1, batch_size=100,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-4,
                },
                verbose=True, print_every=1)
solver.train()