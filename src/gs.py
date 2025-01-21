from typing import List, Optional, Tuple
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

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

pyrootutils.setup_root(
    search_from = __file__,
    indicator = ["environment.yaml"],
    pythonpath= True
)

from src.model.predictor_module import my_predictor_module

device = torch.device("cuda")
alphabet = []

def encode(seq):
    encoded_seq = np.array([alphabet.index(x) for x in seq],dtype=np.float32)
    return encoded_seq

def test_model(pre_net, base_data):
    log.info("Now we will test if the model is correctly resumed")
    features = [torch.from_numpy(encode(x)) for x in base_data.sequence]
    targets = [y for y in base_data.score]

    features = torch.stack(features)  # 将列表中的每个张量堆叠成一个新的张量
    targets = torch.tensor(targets, dtype=torch.float32)  # 转换为 float32 的张量

    # log.info(f"The size of test data is {features.shape}")

    features = features.to(device)
    targets = targets.to(device)

    loss_func = torch.nn.MSELoss()
    with torch.no_grad():
        pred = pre_net(features)
        loss = loss_func(pred, targets)

    # log.info(f"The loss is {loss.item():.4f}")

    if loss <= 0.1:
        log.info("The model is correctly resumed.")
        return 
    else :
        raise ValueError("Resuming failed.")

def generate_a_seq(seq, my_random):
    seq_list = list(seq)
    pos = my_random.randint(0,len(seq_list)-1)
    x = my_random.choice(alphabet)
    seq_list[pos] = x
    return "".join(seq_list)

def run_predictor(pre_net, encoded_seqs, batch_size):
    batchs = torch.split(encoded_seqs, batch_size, 0)
    
    pre_scores = []
    with torch.no_grad():
        for batch in tqdm(batchs, desc="calculate scores", total=len(batchs)):
            pre_scores.append(pre_net(batch))
        pre_scores = torch.concat(pre_scores)
    return pre_scores

def minimum(A, B):
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data > 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)

def levenshtein_knn(x_train, x_test, K, batch_size=1000):
    x_train = x_train.to(device)
    x_test = x_test.to(device)

    def levenshtein_distance(seq1, seq2):
        return torch.sum(seq1 != seq2, dim=-1)

    vals_list = []
    indices_list = []

    for i in tqdm(range(0, x_test.size(0), batch_size), desc="Processing Batches", unit="batch"):
        x_test_batch = x_test[i : i + batch_size]  

        dists = torch.zeros(x_test_batch.size(0), x_train.size(0), device=device)

        for j in range(x_train.size(0)):
            dists[:, j] = levenshtein_distance(x_test_batch, x_train[j])  

        vals, indices = torch.topk(dists, K, dim=1, largest=False) 
        vals_list.append(vals.cpu())
        indices_list.append(indices.cpu())

    vals = torch.cat(vals_list, dim=0)
    indices = torch.cat(indices_list, dim=0)

    return vals.numpy(), indices.numpy()


def generate_evaluate_sequences(cfg, pre_net, base_data):
    raw_seqs = base_data["sequence"].values
    # print(raw_seq[0])
    # print(raw_seq[1])
    # print(raw_seq.shape)
    de_seqs = list(set(raw_seqs))
    uniq_seq = set(de_seqs)
    all_seqs = de_seqs.copy()
    # print(len(de_seq))
    max_n_seqs = cfg.max_n_seqs

    my_random = random.Random(cfg.experiment.random_seed)

    pbar = tqdm(total=max_n_seqs, initial=len(all_seqs), desc="generate samples")

    pointer = 0

    while len(all_seqs) < max_n_seqs:
        new_seq = generate_a_seq(de_seqs[pointer], my_random)
        if new_seq not in uniq_seq:
            all_seqs.append(new_seq)
            uniq_seq.add(new_seq)
            pbar.update(1)
        pointer = (pointer+1)%len(de_seqs)
    pbar.close()

    all_seqs = list(sorted(set(all_seqs)))
    log.info(f"New sequences are generated! There are {len(all_seqs)} samples")
    
    encoded_seqs = [torch.from_numpy(encode(x)) for x in tqdm(all_seqs, desc="encoding", total=len(all_seqs), leave=True)]
    encoded_seqs = torch.stack(encoded_seqs).to(device)

    # print(encoded_seqs.shape)

    pre_scores = run_predictor(pre_net, encoded_seqs, batch_size=5).cpu().numpy()

    log.info(f"Prediction done.Creating KNN graph......")

    vals, indices = levenshtein_knn(encoded_seqs, encoded_seqs, K = int(np.floor(np.sqrt(len(all_seqs)) + 1)), batch_size = 1000)
    # vals, indices = levenshtein_knn(encoded_seqs, encoded_seqs, K = 5, batch_size = 1000)


    log.info(f"KNN graph is done")

    vals = vals[:, 1:]
    indices = indices[:, 1:]

    # print(vals.shape)
    # print(indices.shape)

    # print(vals[:5])
    # print(indices[:5])

    non_mutal_mat = csr_matrix(
        (
        vals.flatten(),
        indices.flatten(),
        np.arange(0, len(vals.flatten())+1, len(vals[0]))
        ),
        shape=(len(vals), len(vals))
        )
    
    # print(non_mutal_mat.shape)
    mutal_mat = minimum(non_mutal_mat, non_mutal_mat.T)

    log.info('Computing Laplacian..')
    L = laplacian(mutal_mat, normed=True).tocsr()    
    return all_seqs, pre_scores, L

    

@hydra.main(version_base=None , config_path="../config" , config_name="gs.yaml")
def main(cfg):
    train_info_dir = cfg.experiment.predictor_dir

    ckpt_path = os.path.join(
        train_info_dir,
        cfg.ckpt_file
    )

    train_config_path = os.path.join(
        train_info_dir,
        "config.yaml"
    )

    with open(train_config_path, 'r') as fp:
        train_config = OmegaConf.load(fp.name)

    # print(OmegaConf.to_yaml(ckpt_cfg, resolve=True))

    task_cfg = train_config.experiment.gfp

    global alphabet
    alphabet = train_config.data.alphabet
    base_data_path = os.path.join(
        task_cfg.task_dir,
        f"mutant_{task_cfg.min_mutant_dist}_percentile_{task_cfg.filter_per[0]}_{task_cfg.filter_per[1]}",
        f"filtered_dataset.csv"
    )

    # print(base_data_path)
    base_data = pd.read_csv(base_data_path)

    log.info(f"The base dataset is loaded, which contains {base_data.shape[0]} samples.")

    # print(OmegaConf.to_yaml(train_config))

    pre_net = my_predictor_module.load_from_checkpoint(
        checkpoint_path = ckpt_path,
        model_cfg = train_config.model
    )
    pre_net.to(device).eval()

    log.info(f"Model parameters have been resumed from the checkpoint.")

    test_model(pre_net, base_data)

    all_seqs, pre_scores, L = generate_evaluate_sequences(cfg, pre_net, base_data)
    gamma = float((cfg.smoothing_method).split('-')[-1])
    # print(f"gamma={gamma}")
    I = identity(L.shape[0], format='csr')
    tmp = I + gamma * L
    y_hat, _ = cg(tmp, pre_scores)
    # print(Y_hat)
    # print(y_hat.shape)
    now = datetime.now().strftime("%m_%d_%Y_%H_%M")

    write_dir = os.path.join(
        train_config.experiment.gfp.task_dir, 
        f"mutant_{train_config.experiment.gfp.min_mutant_dist}_percentile_{train_config.experiment.gfp.filter_per[0]}_{train_config.experiment.gfp.filter_per[1]}",
        cfg.smoothing_method, 
        f"{cfg.exploration_method}_n-{cfg.max_n_seqs // 1000}K"
        )
    os.makedirs(write_dir, exist_ok=True)
    re_write_path = os.path.join(
        write_dir,
        f"re_{now}.csv"
    )
    con_write_path = os.path.join(
        write_dir,
        f"con_{now}.yaml"
    )

    restored_data = pd.DataFrame({
            "sequence": all_seqs,
            "score": y_hat
        })
    
    
    restored_data.to_csv(re_write_path, index=None)
    with open(con_write_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)




if __name__ == "__main__":
    main()