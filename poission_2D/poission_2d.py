import datetime

import torch
from collections import OrderedDict
import numpy as np
import warnings
import time
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader, TensorDataset
from itertools import cycle
import h5py


def read(file_path):
    with h5py.File(file_path, 'r') as hf:
        # 读取 HDF5 文件中的数据集
        dataset = hf['points']
        # 将数据集转换为 NumPy 数组
        data_array = np.array(dataset)
        return data_array


warnings.filterwarnings('ignore')
start_time = time.time()
np.random.seed(1234)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)


# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.SiLU

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]).to(dtype=torch.float64))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]).to(dtype=torch.float64))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


def format_time(t):
    elapsed_rounded = int(round(t))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# the physics-guided neural network


class PhysicsInformedNN:
    def __init__(self, layer, x_data):

        # deep neural networks
        self.dnn = DNN(layer).to(device)
        self.x_data = x_data
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=10000,
            max_eval=10000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
        self.iter = 0

    def net_pde(self, x_in):

        def gd(f, x_1):
            return grad(f, x_1, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

        u = self.dnn(x_in)

        def laplace(f, x1):
            _, col = x1.size()
            u_x = gd(f, x1)
            d_x = u_x[:, 0].unsqueeze(-1)
            d_xx = gd(d_x, x1)[:, 0].unsqueeze(-1)
            for i in range(1, col):
                d_x = u_x[:, i].unsqueeze(-1)
                d_xx += gd(d_x, x1)[:, i].unsqueeze(-1)
            return d_xx

        def f_x(x1):
            row, col = x1.size()
            return 16 * torch.pi ** 2 * col * torch.prod(torch.sin(4 * torch.pi * x1), 1).reshape(-1, 1)

        res = laplace(u, x_in) - f_x(x_in)
        return res

    def loss_func(self, X, Boundary):
        x = X[0].to(device=device)
        x.requires_grad = True
        criterion = torch.nn.MSELoss()
        bc = next(Boundary)[0].to(device=device)
        bc.requires_grad = True
        PDE = self.net_pde(x)
        mse_PDE = criterion(PDE, torch.zeros_like(PDE))
        BC = self.p_loss(bc)
        mse_BC = criterion(BC, torch.zeros_like(BC))
        p = self.p_loss(x)
        mse_p = criterion(p, torch.zeros_like(p))
        loss = mse_BC + mse_PDE + mse_p
        self.iter += 1
        if self.iter % 500 == 0:
            print('iter: %d, Loss: %e' % (self.iter, loss.item()))
        return loss

    def p_loss(self, x_in):
        y = self.dnn(x_in) + torch.prod(torch.sin(4 * torch.pi * x_in), 1).reshape(-1, 1)
        return y

    def train(self, nIter, X, Boundary):
        self.dnn.train()
        t0 = time.time()
        for epoch in range(nIter):
            for x_data in X:
                x = x_data[0].to(device=device)
                x.requires_grad = True
                criterion = torch.nn.MSELoss()
                bc = next(Boundary)[0].to(device=device)
                bc.requires_grad = True
                PDE = self.net_pde(x)
                mse_PDE = criterion(PDE, torch.zeros_like(PDE))
                BC = self.p_loss(bc)
                mse_BC = criterion(BC, torch.zeros_like(BC))
                p = self.p_loss(x)
                mse_p = criterion(p, torch.zeros_like(p))
                loss = mse_BC + 10 * mse_PDE + mse_p
                # Backward and optimize
                self.optimizer_Adam.zero_grad()
                loss.backward()
                self.optimizer_Adam.step()
            if epoch % 500 == 0:
                t1 = time.time()
                print(
                    'It: %d, Loss: %.3e, pde: %.3e, bc: %.3e, time: %s' %
                    (
                        epoch,
                        loss.item(),
                        mse_PDE.item(),
                        mse_BC.item(),
                        format_time(t1 - t0)
                    )
                )
                t0 = t1

        def closure():
            self.optimizer.zero_grad()
            l_loss = self.loss_func(lbfgs_x, Boundary)
            l_loss.backward()
            return l_loss

        i = 0
        for lbfgs_x in self.x_data:
            self.iter = 0
            self.optimizer.step(closure)
            loss = closure()
            i += 1
            print('batch: %d, Loss: %e' % (i, loss.item()))

    def predict(self, X):
        x = torch.tensor(X, requires_grad=True).float().to(device)
        self.dnn.eval()
        y = self.dnn(x)
        y = y.detach().cpu().numpy()
        return y


dimension = 2
d_nums = 200
layers = [dimension, d_nums, d_nums, d_nums, d_nums, d_nums, d_nums, 1]

x_data_path = 'points_2d.h5'
boundary_path = 'p_2d.h5'
# 创建数据集实例
data = read(x_data_path)
tensor_data = torch.tensor(data, dtype=torch.float64)
print(len(data))
batch_size = len(data)
batch_size_2 = len(data)
# 创建一个 TensorDataset

x_set = TensorDataset(tensor_data)
x_gen = DataLoader(x_set, batch_size=batch_size, shuffle=True)
x_gen_2 = DataLoader(x_set, batch_size=batch_size_2, shuffle=True)

boundary = read(boundary_path)
boundary_set = TensorDataset(torch.tensor(boundary, dtype=torch.float64))
boundary_gen = cycle(DataLoader(boundary_set, batch_size=batch_size, shuffle=True))

# training
model = PhysicsInformedNN(layers, x_gen_2)
model.dnn = model.dnn
model.train(8000, x_gen, boundary_gen)
#
# 保存模型的状态字典
torch.save(model.dnn.state_dict(), 'model_2D.pth')
