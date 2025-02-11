import os.path

import pandas as pd
import requests
from tqdm import tqdm


import matplotlib.pyplot as plt
from collections import Counter


def get_protein_solubility(sequence):
    """
    :param sequence: protein sequence
    :return: predicted solubility
    """
    url = "https://www.novopro.cn/plus/ppc.php"
    headers = {
        "Cookie": "_pk_id.1.24a9=0642b0864d52ba7c.1704679236.; _pk_ses.1.24a9=1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    data = {
        "sr": "sol",
        "sq": sequence
    }
    r = requests.post(url, headers=headers, data=data)
    return r.text


def get_sol_entrance(sequences):
    if isinstance(sequences, str):
        pred = get_protein_solubility(sequences)
        pred = float(pred.split(',')[1][:len(pred.split(',')[1])-1])
        return pred

    pred_sol = []
    for i,sequence in tqdm(enumerate(sequences),desc="calc pred_sol",total=len(sequences)):
    # for i,sequence in enumerate(sequences):
        pred = get_protein_solubility(sequence)
        pred = float(pred.split(',')[1][:len(pred.split(',')[1])-1])
        pred_sol.append(pred)

    return pred_sol

# seqs = pd.read_csv("data/GFP/mutant_7_percentile_0.0_0.3/filtered_dataset.csv")["sequence"]
# seqs = seqs[:10]
# # seqs = pd.read_csv("data/GFP/ground_truth.csv")["sequence"]
# data,lenn = get_sol_entrance(seqs)

# print(lenn)

# counter = Counter(data)
# numbers = list(counter.keys())
# frequencies = list(counter.values())

# # 绘制散点图
# plt.scatter(numbers, frequencies, color='blue', alpha=0.6)
# plt.title('Pred-Solubility Frequency Scatter Plot')
# plt.xlabel('Solubility')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()
