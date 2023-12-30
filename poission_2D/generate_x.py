import h5py
import numpy as np

# 定义每个维度上的坐标点数量和范围


dimension = 2
nums = 100 ** 2
batch_size = 10000
min_range = -1
max_range = 1

# 生成多维空间中格点的坐标
output_file = 'points_2d.h5'
with h5py.File(output_file, 'w') as file:
    # 创建一个数据集
    dataset = file.create_dataset('points', shape=(nums, dimension), dtype=np.float64)

    # 逐批次生成坐标点并写入数据集
    for i in range(0, nums, batch_size):
        batch_points = np.random.uniform(min_range, max_range, (batch_size, dimension))
        dataset[i:i + batch_size] = batch_points

print("坐标点生成完成并已保存到文件:", output_file)
