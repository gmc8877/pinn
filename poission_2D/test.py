import numpy as np
import torch
from collections import OrderedDict


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
        out = -torch.prod(torch.sin(4 * torch.pi * x), 1).reshape(-1, 1)
        return out


dimension = 2
d_nums = 200
layers = [dimension, d_nums, d_nums, d_nums, d_nums, d_nums, d_nums, 1]

# 输入n个测试点，形状为n*10的矩阵
# 例如
dimension = 2
nums = 100
min_range = -1
max_range = 1
points = np.random.uniform(min_range, max_range, (nums, dimension))


# 预测函数
def predict(x):
    tensor_x = torch.tensor(x, dtype=torch.float64)
    model = DNN(layers)
    model.load_state_dict(torch.load('model_2D.pth'))
    u_pred = model(tensor_x).detach().cpu().numpy()
    return u_pred


# 保存预测数据

y_pred = predict(points)

np.save('y_pred.npy', y_pred)
