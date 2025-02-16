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


from utils import get_sol_entrance, get_ogt_entrance, get_ogt_model
from src.model.predictor_module import my_predictor_module

device = torch.device("cuda")

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





@hydra.main(version_base=None , config_path="../config" , config_name="gs.yaml")
def main(cfg):
    wt_seq = "SKGEELSTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKSEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYILADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    # print(get_sol_entrance(wt_seq))
    my_random = random.Random(cfg.experiment.random_seed)
    raw_data = pd.read_csv("data/GFP/mutant_7_percentile_0.0_0.3/filtered_dataset_with_ogtsol.csv")
    sequences = [x for x in raw_data.sequence]
    for _ in range(cfg.epoch):
        sequences = mutate(sequences, cfg.max_n_seqs, my_random)












if __name__=="__main__":
    main()