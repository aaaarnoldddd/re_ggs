import hydra
import numpy as np
import pandas as pd
from random import sample
import random
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix, load_npz
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pyrootutils
import torch
from copy import deepcopy
import logging
import time
import os
from datetime import datetime
from scipy.sparse.linalg import cg
from scipy.sparse import identity, csr_matrix, save_npz
from tqdm import tqdm
import sys
import pickle as pkl
from collections import defaultdict

import matplotlib.pyplot as plt
from collections import Counter

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

pyrootutils.setup_root(
    search_from = __file__,
    indicator = ["environment.yaml"],
    pythonpath= True
)


from utils import get_sol_entrance, get_ogt_entrance
from src.model import MTL_Module

device = torch.device("cuda")

def encode(seq):
    alphabet = "ARNDCQEGHILKMFPSTWYV"
    encoded_seq = np.array([alphabet.index(x) for x in seq],dtype=np.float32)
    return encoded_seq

def mutate(sequences, max_n_seqs, my_random):
    alphabet = "ARNDCQEGHILKMFPSTWYV"
    tot = 0
    set_seqs = set(sequences)
    new_seqs = []
    while tot<max_n_seqs:
        x = my_random.randint(0, len(sequences)-1)
        pos = my_random.randint(0, len(sequences[0])-1)
        to = my_random.choice(alphabet)
        new_seq = list(sequences[x])
        new_seq[pos] = to
        new_seq = ''.join(new_seq)
        if new_seq not in set_seqs:
            set_seqs.add(new_seq)
            tot+=1
            new_seqs.append(new_seq)
    return new_seqs

def get_preds(sequences, batch_size):
    ckpt_path = "ckpt/GFP/mutant_7/percentile_0.0_0.3/unsmoothed/02_16_2025_16_35/last.ckpt"
    train_config_path = "ckpt/GFP/mutant_7/percentile_0.0_0.3/unsmoothed/02_16_2025_14_36/config.yaml"

    with open(train_config_path, 'r') as fp:
        train_config = OmegaConf.load(fp.name)
    num_tasks = train_config.model.mtl.num_tasks
    predictor = MTL_Module.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        mcfg=train_config.model.mtl,
        ocfg=train_config.model.optimizer
    )
    predictor.to(device).eval()
    log.info(f"Model parameters have been resumed from the checkpoint.")

    encoded_seqs = [torch.from_numpy(encode(x)) for x in tqdm(sequences, desc="encoding", total=len(sequences), leave=True)]
    encoded_seqs = torch.stack(encoded_seqs).to(device)

    batchs = torch.split(encoded_seqs, batch_size, 0)

    pre_scores = []
    with torch.no_grad():
        for batch in tqdm(batchs, desc="calculate scores", total=len(batchs)):
            tmp = predictor(batch)
            pre_scores.append(torch.concat([tmp[f"task{i+1}_pred"] for i in range(num_tasks)], dim=1))
        pre_scores = torch.concat(pre_scores)

    return pre_scores

def process(pred_sol, pred_ogt, wt_sol, wt_ogt, alpha=0.5, beta=0.5):
    """
    返回:
      pred_c: np.array(shape=(N,)), 每条序列的副目标代价, 越小越好
    """
    eps = 1e-8

    sol_min, sol_max = pred_sol.min(), pred_sol.max()
    norm_sol = (pred_sol - sol_min) / (max(sol_max - sol_min, eps))

    ogt_diff = np.abs(pred_ogt - wt_ogt)
    diff_min, diff_max = ogt_diff.min(), ogt_diff.max()
    norm_ogt_diff = (ogt_diff - diff_min) / (max(diff_max - diff_min, eps))

    pred_c = alpha * (1 - norm_sol) + beta * norm_ogt_diff

    return pred_c

def hamming_distance(seq1, seq2):
    return sum(ch1 != ch2 for ch1, ch2 in zip(seq1, seq2))
    
def verify_and_plot_frontier(sequences, wt_seq, pred_score, query_batch):
    """
    绘制所有候选序列与从 get_convex(...) 得到的 query_batch 在
    (汉明距离, 预测得分) 平面的分布，方便直观验证前沿效果。
    
    参数:
      sequences  : list of str, 所有候选序列（与 pred_score 一一对应）
      wt_seq     : str, 野生型序列
      pred_score : np.array, shape=(N,), 对应所有候选序列的模型预测得分
      query_batch: list of str, 来自 get_convex(...) 的前沿序列集合
    """
    # 1. 计算所有候选序列的 (distance, score)
    dist_all = []
    score_all = []
    for i, seq in enumerate(sequences):
        d = sum(a != b for a, b in zip(seq, wt_seq))  # hamming distance
        dist_all.append(d)
        score_all.append(pred_score[i])
    
    # 2. 为了找出 query_batch 中点的 (distance, score)，
    #    需要匹配它们在 sequences 里的位置 (考虑重复序列情况)
    #    用一个 dict: seq -> list of [indices], 以便逐个匹配
    seq_map = defaultdict(list)
    for i, seq in enumerate(sequences):
        seq_map[seq].append(i)
    
    used_indices = set()   # 防止同一个序列多次匹配到相同 index
    dist_query = []
    score_query = []
    
    for qb_seq in query_batch:
        # 找 seq_map[qb_seq] 中第一个未被使用的 index
        if qb_seq not in seq_map:
            # 说明前沿里出现了一个不在原 sequences 列表的序列(很少见)
            continue
        found_idx = None
        for idx in seq_map[qb_seq]:
            if idx not in used_indices:
                found_idx = idx
                break
        if found_idx is not None:
            used_indices.add(found_idx)
            # 记录该前沿点的 distance & score
            d_q = sum(a != b for a, b in zip(qb_seq, wt_seq))
            dist_query.append(d_q)
            score_query.append(pred_score[found_idx])
    
    # 3. 绘制散点图
    plt.figure(figsize=(8,6))
    # 所有候选点 (蓝色)
    plt.scatter(dist_all, score_all, c='blue', alpha=0.5, label='All candidates')
    # 前沿序列 (红色 "x")
    plt.scatter(dist_query, score_query, c='red', marker='x', s=100, label='Query batch(frontier)')
    
    plt.xlabel('Hamming distance to WT')
    plt.ylabel('Predicted score')
    plt.title('All sequences vs. Frontier (Query Batch)')
    plt.legend()
    plt.show()

def get_convex(sequences, wt_seq, pred_score, sol, ogt, 
               batch_size=128, num_queries=10):
    """
    仿照你给的伪代码，只取“半个凸包”作为frontier。
    同时返回选中序列对应的 sol 和 ogt。

    参数：
      sequences   : list of sequences（候选序列列表）
      wt_seq      : 野生型序列（字符串）
      pred_score  : np.array, shape (N,)，每个序列的主目标得分（例如模型预测适应度）
      sol         : np.array, shape (N,)，每条序列的溶解度
      ogt         : np.array, shape (N,)，每条序列的 OGT
      batch_size  : int，每次处理多少序列（与示例伪代码保持对应）
      num_queries : int，每轮希望选出的序列数量 (self.num_queries_per_round)
    
    返回：
      query_batch : list of sequences（被选中的序列，依照“前沿”提取）
      query_sol   : list of float，与 query_batch 对应的溶解度
      query_ogt   : list of float，与 query_batch 对应的 OGT
    """

    # ------------------------------
    # 1) 构造 candidate_pool_dict
    #    键：distance_to_wt； 值： [{sequence, model_score, sol, ogt}, ...]（按得分降序）
    N = len(sequences)
    candidate_pool_dict = {}

    # 按 batch_size 分批处理
    for i in range(0, N, batch_size):
        candidate_batch = sequences[i : i + batch_size]
        score_batch = pred_score[i : i + batch_size]
        sol_batch   = sol[i : i + batch_size]
        ogt_batch   = ogt[i : i + batch_size]

        # 将序列根据 distance_to_wt 分组
        for seq, sc, so, og_ in zip(candidate_batch, score_batch, sol_batch, ogt_batch):
            dist = sum(a != b for a, b in zip(seq, wt_seq))  # hamming_distance
            if dist not in candidate_pool_dict:
                candidate_pool_dict[dist] = []
            candidate_pool_dict[dist].append(dict(
                sequence=seq, 
                model_score=sc, 
                sol=so, 
                ogt=og_
            ))
    
    # 对字典中每个距离组内的列表按 model_score 降序排序
    for dist in sorted(candidate_pool_dict.keys()):
        candidate_pool_dict[dist].sort(key=lambda x: x['model_score'], reverse=True)

    # ------------------------------
    # 2) 构造 query_batch by iteratively extracting the proximal frontier
    query_batch = []
    query_sol   = []
    query_ogt   = []

    while len(query_batch) < num_queries:
        # Compute the proximal frontier by Andrew's monotone chain
        stack = []
        
        # 按距离从小到大遍历
        sorted_dists = sorted(candidate_pool_dict.keys())
        for dist in sorted_dists:
            if len(candidate_pool_dict[dist]) > 0:
                data = candidate_pool_dict[dist][0]  # 该距离组中得分最高的项
                new_point = np.array([dist, data['model_score']])
                
                # check_convex_hull: <= 0 表示p3在p1->p2叉积的非逆时针侧 => pop
                def check_convex_hull(p1, p2, p3):
                    return np.cross(p2 - p1, p3 - p1) <= 0

                while len(stack) > 1 and not check_convex_hull(stack[-2], stack[-1], new_point):
                    stack.pop(-1)
                stack.append(new_point)
        
        # 再检查末尾，如果最后一个点在 y 轴(得分)上低于倒数第二个，弹出
        while len(stack) >= 2 and stack[-1][1] < stack[-2][1]:
            stack.pop(-1)
        
        # ------------------------------
        # 3) Update query batch and candidate pool.
        for (dist, sc) in stack:
            dist = int(dist)  # 字典键
            if len(query_batch) < num_queries:
                if candidate_pool_dict[dist]:
                    chosen = candidate_pool_dict[dist][0]  # dict(sequence, model_score, sol, ogt)
                    
                    query_batch.append(chosen['sequence'])
                    query_sol.append(chosen['sol'])
                    query_ogt.append(chosen['ogt'])
                    
                    # 从该距离组移除
                    candidate_pool_dict[dist].pop(0)
            else:
                break

        # 如果所有池子都空了，就 break
        empty_check = all(len(lst) == 0 for lst in candidate_pool_dict.values())
        if empty_check:
            break

    return query_batch, np.array(query_sol), np.array(query_ogt)

def select_batch(query_batch, pred_c, T_init=1.0, n_iter=1000, cooling=None, return_mode='chain'):
    """
    对 query_batch 做 Metropolis MCMC 策略的“拒绝–接受”筛选。
    相比一次性抽样，这里通过多轮迭代(随机提案)来保留更多多样性。

    参数:
      query_batch : list of sequences
        来自上一阶段 (get_convex(...)) 的候选序列
      pred_c      : np.array, shape=(M,), 与 query_batch 对应的副目标代价(越小越好)
      T_init      : float, 初始温度(越大越容易接受更差解)
      n_iter      : int,   迭代步数
      cooling     : float 或 None, 若不为 None, 每步后 T = T * cooling (模拟退火)
      return_mode : str,   取值可为:
          'chain'   => 返回整个 MCMC 采样链(list of seq)
          'best'    => 返回其中代价最优的那条序列
          'last'    => 返回链中最后一次接受的序列

    返回:
      new_seqs : list of sequences (如果 return_mode='chain' )
                 或 单个 sequence (如果 return_mode='best' / 'last')

    用法示例:
        new_chain = select_batch(query_batch, pred_c, T_init=2.0, n_iter=2000, cooling=0.99)
        best_sol  = select_batch(query_batch, pred_c, return_mode='best')
    """

    # 1) 先把 query_batch 和 pred_c 合并到一个列表 S，便于随机索引
    if len(query_batch) != len(pred_c):
        raise ValueError("query_batch 与 pred_c 长度不匹配!")
    
    S = list(zip(query_batch, pred_c))  # [(seq, cost), ...]
    M = len(S)
    if M == 0:
        return [] if return_mode == 'chain' else None

    # 2) 随机初始化 (从 S 中任意选一个做当前解)
    np.random.seed()  # 这里可根据需求设置固定种子
    idx = np.random.randint(M)
    current_seq, current_cost = S[idx]

    best_seq, best_cost = current_seq, current_cost
    chain = [(current_seq, current_cost)]  # 记录所有采样点

    T = T_init

    # 3) 迭代
    for it in range(n_iter-1):
        # 随机提案
        new_idx = np.random.randint(M)
        new_seq, new_cost = S[new_idx]

        delta = new_cost - current_cost
        if delta < 0:
            # 更优 => 必接收
            current_seq = new_seq
            current_cost = new_cost
        else:
            # 更差 => 以 e^(-delta/T) 概率接受
            prob = np.exp(-delta / T)
            if np.random.rand() < prob:
                current_seq = new_seq
                current_cost = new_cost
        
        # 更新全局最好
        if current_cost < best_cost:
            best_seq = current_seq
            best_cost = current_cost

        # 记录
        chain.append((current_seq, current_cost))

        # 温度衰减 (若需要)
        if cooling is not None:
            T = T * cooling

    # 4) 根据 return_mode 返回
    if return_mode == 'chain':
        # 返回整条链: list of seq
        # chain 中每元素是 (seq, cost)，我们只取 seq
        return [x[0] for x in chain]
    elif return_mode == 'best':
        # 返回整轮采样中代价最低的那个序列
        return best_seq
    elif return_mode == 'last':
        # 返回链的最后一次采样
        return chain[-1][0]
    else:
        raise ValueError(f"未知 return_mode: {return_mode}")




@hydra.main(version_base=None , config_path="../config" , config_name="gs.yaml")
def main(cfg):
    wt_seq = "SKGEELSTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKSEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYILADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    # print(get_sol_entrance(wt_seq))
    wt_sol = np.float32(get_sol_entrance(wt_seq))
    wt_ogt = np.float32(get_ogt_entrance(wt_seq))

    my_random = random.Random(cfg.experiment.random_seed)
    raw_data = pd.read_csv("data/GFP/mutant_7_percentile_0.0_0.3/filtered_dataset_with_ogtsol.csv")
    sequences = [x for x in raw_data.sequence]
    for _ in range(cfg.epoch):
        sequences = mutate(sequences, cfg.max_n_seqs, my_random)
        preds = get_preds(sequences, batch_size=128).cpu().numpy()

        pred_score = np.array([x[0] for x in preds])
        pred_sol = np.array([x[1] for x in preds])
        pred_ogt = np.array([x[2] for x in preds])

        query_batch, pred_sol, pred_ogt = get_convex(sequences, wt_seq, pred_score, pred_sol, pred_ogt, batch_size=64, num_queries=120)

        pred_c = process(pred_sol, pred_ogt, wt_sol, wt_ogt)
        
        selected_batch = select_batch(query_batch, pred_c)

        # new_seqs = selecet_batch(query_batch, pred_c)

        # verify_and_plot_frontier(sequences, wt_seq, pred_score, query_batch)
        x=5











if __name__=="__main__":
    main()