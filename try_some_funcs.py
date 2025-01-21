import torch

import pandas as pd
import os
import pyrootutils
from tqdm import tqdm
import random
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

pyrootutils.setup_root(
    search_from = __file__,
    indicator = ["environment.yaml"],
    pythonpath= True
)

# data = pd.read_csv("data/GFP/ground_truth.csv")
# print(data.shape[0])
# x = data.score.quantile(0.999)
# y = data.score.quantile(0.900)
# print(x,y)

# data=data.reset_index(drop=True)

# nnn = [(x,y) for x,y in zip(data.sequence, data.score)]

# # nnn=nnn.reset_index(drop=True)

# sequence = data.sequence

# for x in nnn:
#     print(x)

# sequence = sequence.tolist()

# print(type(nnn))

# df = pd.DataFrame({
#             "sequence": [seq for seq, _ in nnn],
#             "score": [score for _, score in nnn]
#         })

# print(df)
# print(data)
# # print(nnn[4])

# tot = sum(1 for (_, x) in nnn if x < 0)

# print(tot)


# print(f"hi")
# pykeops.test_torch_bindings()    # perform the compilation


# def levenshtein_knn(x_train, x_test, K, batch_size=1000):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x_train = x_train.to(device)
#     x_test = x_test.to(device)

#     def levenshtein_distance(seq1, seq2):
#         return torch.sum((seq1-seq2)**2, dim=-1)

#     vals_list = []
#     indices_list = []

#     # 按照分块处理测试数据
#     for i in tqdm(range(0, x_test.size(0), batch_size), desc="Processing Batches", unit="batch"):
#         x_test_batch = x_test[i : i + batch_size]  # 取测试数据的一个分块

#         # 初始化分块距离矩阵
#         dists = torch.zeros(x_test_batch.size(0), x_train.size(0), device=device)

#         # 逐个计算 Levenshtein 距离
#         for j in range(x_train.size(0)):
#             dists[:, j] = levenshtein_distance(x_test_batch, x_train[j])  # x_train[j] 是一个训练点

#         # 提取最近邻
#         vals, indices = torch.topk(dists, K, dim=1, largest=False)  # 取最近 K 个邻居
#         vals_list.append(vals.cpu())
#         indices_list.append(indices.cpu())

#     # 合并所有分块结果
#     vals = torch.cat(vals_list, dim=0)
#     indices = torch.cat(indices_list, dim=0)

#     return vals, indices

# # 测试代码
# if __name__ == "__main__":
#     # 训练数据和测试数据
#     N, M, D, K = 10, 10, 2, 3  # 训练点数、测试点数、维度、最近邻个数
#     x_train = torch.randint(0,10,(N,D))
#     x_test = torch.randint(0,10,(M, D))

#     print(x_train)
#     print(x_test)

#     print(numpy.arange(0,7))

    # 运行 KNN
    # vals, indices = levenshtein_knn(x_train, x_train, K, batch_size=2000)

    # print(f"最近邻距离形状: {vals.shape}")  # [M, K]
    # print(f"最近邻索引形状: {indices.shape}")  # [M, K]

    # print(vals)
    # print(indices)

# def generate_a_seq(seq):
#     seq_list = list(seq)
#     pos = random.randint(0,5)
#     x = random.choice("ARNDCQEGHILKMFPSTWYV")
#     print(f"pos is {pos}, change to {x}")
#     seq_list[pos] = x
#     return "".join(seq_list)
#     # return seq_list

# x = [1,1,2,3,4,4]
# y = list(set(x))

# # print(len(x),len(y))

# # print(torch.randint(0,10,(1,1)).item())
# print("fuck")


data = np.random.randint(low=0, high=10, size=(10,2))
print('\n'.join([f"{id, c}" for id, c in enumerate(data)]))

Z = linkage(data, method='average')

plt.figure(figsize=(8,5))
dendrogram(Z)
plt.show()

labels = fcluster(Z, t=5, criterion='distance')
print(labels)
# breakpoint()