import h5py
import numpy as np

dimension = 10
nums_0 = 1000000
nums_1 = 8000000
batch_size = 1000000
min_range = -1
max_range = 1


# 生成多维空间中格点的坐标
output_file = 'p_10d.h5'

with h5py.File(output_file, 'w') as file:
    # 创建一个数据集
    dataset = file.create_dataset('points', shape=(nums_0 + nums_1, dimension), dtype=np.float64)
    # 逐批次生成坐标点并写入数据集
    for i in range(0, nums_0, batch_size):
        point = np.random.randint(min_range * 8, max_range * 8, (batch_size, dimension))
        point = point / 8
        dataset[i:i + batch_size] = point
    for i in range(0, nums_1, batch_size):
        point = np.random.randint(min_range * 4, max_range * 4, (batch_size, dimension))
        point = (point*2+1)/8
        dataset[nums_0 + i:i + batch_size] = point
print("坐标点生成完成并已保存到文件:", output_file)
