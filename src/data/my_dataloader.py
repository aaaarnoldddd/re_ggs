from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import logging
import os
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler,random_split
import logging
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from utils import get_sol_entrance, get_ogt_entrance, get_ogt_model

class my_data_module(LightningDataModule):
    def __init__(
            self,
            *,
            task_cfg,
            task,
            batch_size,
            num_workers,
            seed,
            alphabet,
            sequence_column = "sequence",
            weighted_sampling,
    ):
        super().__init__()
        self._task_cfg=task_cfg
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._seed = seed
        self._weighted_sampling = weighted_sampling
        self._alphabet = alphabet
        self._log = logging.getLogger(__name__)
        self._pin_memory=self._task_cfg.pin_memory
        self._smoothing_params = self._task_cfg.smoothing_params
        self._val_ratio = self._task_cfg.val_ratio
        self._test_ratio = self._task_cfg.test_ratio


    def _encode(self,seq):
        encoded_seq = np.array([self._alphabet.index(x) for x in seq])
        return encoded_seq

    def _decode(self,seq):
        decoded_seq = "".join(self._alphabet[x] for x in seq)
        return decoded_seq

    def _get_min_dist(self,seq):
        ret = len(seq)
        for t in self._tops.sequence:
            s = sum(1 for x,y in zip(t,seq) if x!=y)
            ret = min(ret, s)
        return ret

        
    def setup_unsmoothed(self):
        raw_data = pd.read_csv(self._task_cfg.csv_path)

        self._log.info(f"get data from {self._task_cfg.csv_path}")

        raw_nums = raw_data.shape[0]
        top_quantile = self._task_cfg.top_quantile
        filter_per = [self._task_cfg.filter_percentile[0],self._task_cfg.filter_percentile[1]]
        min_mutant_dist = self._task_cfg.min_mutant_dist

        self._tops = raw_data[raw_data.score >= raw_data.score.quantile(top_quantile)]

        self._log.info(f"The data between {filter_per[0]*100}% and {filter_per[1]*100}% will be filterer out, which is between {raw_data.score.quantile(filter_per[0])} and {raw_data.score.quantile(filter_per[1])}")

        filtered = raw_data[raw_data.score.between(raw_data.score.quantile(filter_per[0]), raw_data.score.quantile(filter_per[1]))]

        # filtered = filtered[:50]

        new_data = [(x, np.float32(y)) 
                    for x,y in 
                    tqdm(zip(filtered.sequence, filtered.score), desc="Processing sequences", total=len(filtered),leave=True)
                    if self._get_min_dist(x) >= min_mutant_dist]

        pred_sol = get_sol_entrance([x[0] for x in new_data])
        pred_ogt = get_ogt_entrance([x[0] for x in new_data])

        self._dataset = [(self._encode(x[0]), (x[1], np.float32(y), np.float32(z)))
                         for x, y, z in
                         zip(new_data, pred_sol, pred_ogt)]
        
        # self._dataset = self._dataset[:100]
        
        self._log.info("Dataset done")
        self._log.info(f"{len(self._dataset)} samples has been filtered out")

        write_dir = os.path.join(self._task_cfg.task_dir, f"mutant_{self._task_cfg.min_mutant_dist}_percentile_{self._task_cfg.filter_percentile[0]}_{self._task_cfg.filter_percentile[1]}")
        os.makedirs(write_dir, exist_ok=True)
        write_path = os.path.join(write_dir, f"filtered_dataset_with_ogtsol.csv")

        df = pd.DataFrame({
            "sequence": [self._decode(seq) for seq, _ in self._dataset],
            "score": [x[0] for _, x in self._dataset],
            "sol": [x[1] for _, x in self._dataset],
            "OGT": [x[2] for _, x in self._dataset]
        })
                
        self._log.info(f"Write the dataset to {write_path}")
        df.to_csv(write_path, index=False)
        self._log.info(f"training size is {len(self._dataset)}")

    def setup_smoothed(self):
        raw_data = pd.read_csv(self._task_cfg.csv_path)

        self._dataset = [(self._encode(x),np.float32(y)) for x,y in zip(raw_data.sequence, raw_data.score)]
        
        self._log.info(f"get data from {self._task_cfg.csv_path}")
        self._log.info(f'Read in {len(self._dataset)} smoothed sequences.')

    def setup_(self):
        raw_data = pd.read_csv("data/GFP/mutant_7_percentile_0.0_0.3/filtered_dataset_with_ogtsol.csv")
        self._dataset = [(self._encode(x), (np.float32(y), np.float32(z), np.float32(p))) 
                        for x,y,z,p
                        in zip(raw_data.sequence, raw_data.score, raw_data.sol, raw_data.OGT)]
        
        # self._dataset = self._dataset[:10]
        self._log.info(f'Read in {len(self._dataset)} smoothed sequences.')

    def setup(self, stage=None):
        self._log.info("Start preparing dataset")


        if self._smoothing_params == 'unsmoothed':
            # self.setup_unsmoothed()
            self.setup_()
    
        else:
            self.setup_smoothed()

        dataset_len = len(self._dataset)
        val_len = int(self._val_ratio * dataset_len)
        test_len = int(self._test_ratio * dataset_len)
        train_len = dataset_len - val_len - test_len

        self._train_dataset, self._val_dataset, self._test_dataset = random_split(
            self._dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self._seed)
        )

        self._log.info(f"Total samples = {dataset_len}, "
                       f"train = {train_len}, val = {val_len}, test = {test_len}")



        self._log.info(f"Use weighted sampling")
        targets = [item[1] for item in self._train_dataset]  # (x[1], y, z)
        targets = np.array(targets)  # 转换为 numpy 数组，形状 (N, 3)

        # 方法1: 取 target 平均值
        # combined_targets = targets.mean(axis=1)  

        # 方法2: 取最大值
        # combined_targets = targets.max(axis=1)

        # 方法3: 自定义加权平均 (调整 a, b, c)
        a, b, c = 0.6, 0.2, 0.2  # 你可以调整权重
        sum = targets.sum(axis=0)
        targets = targets/sum
        combined_targets = a * targets[:, 0] + b * targets[:, 1] + c * targets[:, 2]

        # 计算采样权重
        adjusted_targets = combined_targets - combined_targets.min() + 1  # 避免除 0
        weights = 1 / adjusted_targets  # 目标值越大，权重越小（即越容易被采样）

        # 创建 WeightedRandomSampler
        self._sampler = WeightedRandomSampler(weights, len(weights))

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size = self._batch_size,
            num_workers = self._num_workers,
            pin_memory = self._pin_memory,
            sampler = self._sampler
        )

    def val_dataloader(self):
        """
        通常验证集/测试集只需要 SequentialSampler (默认) 即可
        """
        if self._val_dataset is None or len(self._val_dataset) == 0:
            return None  
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory
        )
    
    def test_dataloader(self):
        if self._test_dataset is None or len(self._test_dataset) == 0:
            return None
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory
        )
    


class my_dataset(Dataset):
    def __init__(
            self,
            *,
            csv_path,
            cluster_cutoff,
            max_visits,
            clustering: True
            ):
        self._log = logging.getLogger(__name__)
        self._log.info(f"Reading csv file from {csv_path}")
        self._raw_data = pd.read_csv(csv_path)
        self._data = self._raw_data.copy()
        self._log.info(
            f"Found {len(self.sequences)} sequences "
            f"with TRUE scores between {np.min(self.scores):.2f} and {np.max(self.scores):.2f}"
        )
                 
        self._cluster_cutoff = cluster_cutoff
        
        self._observed_sequences = {seq: 1 for seq in self.sequences}
        self._max_visits = max_visits
        self._pairs = pd.DataFrame({
            'source_sequence': [],
            'mutant_sequence': [],
            'source_score': [],
            'mutant_score': [],
            'epoch': [],
        })

        self._cluster_centers = self._pairs.copy() if clustering else None
        self.cluster()
    
    @property
    def sequences(self):
        return self._data.sequence.tolist()
    @property
    def scores(self):
        return self._data.score.tolist()
    @property
    def pairs(self):
        return self._pairs
    
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx): 
        row = self._data.iloc[idx]
        return {
            'sequence': row['sequence'],
            'score': row['score'],
        }
    
    def add_pairs(self, new_pairs, epoch):
        prev_num_pairs = len(self._pairs)
        new_pairs['epoch'] = epoch
        updated_pairs = pd.concat([self._pairs, new_pairs])
        updated_pairs = updated_pairs.drop_duplicates(
            subset=['source_sequence', 'mutant_sequence'], ignore_index=True)
        num_new_pairs = len(updated_pairs) - prev_num_pairs
        self._log.info(f'Added {len(updated_pairs) - prev_num_pairs} pairs.')
        self._pairs = updated_pairs
        return num_new_pairs

    def get_visits(self, sequences):
        return [
            self._observed_sequences[seq] if seq in self._observed_sequences else 0
            for seq in sequences
        ]

    def cluster(self):
        alphabet = "ARNDCQEGHILKMFPSTWYV"
        seq_ints = [
            [alphabet.index(c) for c in seq] for seq in self.sequences
        ]
        seq_array = np.array(seq_ints)

        Z = linkage(seq_array, method='average', metric='hamming')
        cluster_assignments = fcluster(Z, t=self._cluster_cutoff, criterion='maxclust')

        prev_num_seqs = len(self.sequences)
        self._data['cluster'] = cluster_assignments.tolist()
        max_cluster_fitness = {}
        for cluster, cluster_df in self._data.groupby('cluster'):
            max_cluster_fitness[cluster] = cluster_df['score'].max()
        self._data = self._data[
            self._data.apply(
                lambda x: x.score == max_cluster_fitness[x.cluster], axis=1
            )
        ]
        self._cluster_centers = pd.concat([self._cluster_centers, self._data])
        self._log.info(
            f"Clustered {prev_num_seqs} sequences to {len(self.sequences)} sequences "
            f"with scores min={np.min(self.scores):.2f}, max={np.max(self.scores):.2f}, "
            f"mean={np.mean(self.scores):.2f}, std={np.std(self.scores):.2f}"
        )

    def remove(self, seqs):
        """Remove sequence(s) and score(s)."""
        if not isinstance(seqs, list):
            seqs = [seqs]
        if len(seqs) == 0:
            return
        prev_num_seqs = len(self.sequences)
        self._data = self._data[~self._data.sequence.isin(seqs)]
        removed_num_seqs = len(self.sequences) - prev_num_seqs
        self._log.info(f"Removed {removed_num_seqs} sequences.")

    def reset(self):
        self._data = pd.DataFrame(columns=self._data.columns)

    def add(self, new_seqs):
        """Add sequence(s) and score(s) to the end of the dataset"""
        filtered_seqs = new_seqs[np.array(self.get_visits(new_seqs.sequence)) < self._max_visits]
        prev_num_seqs = len(self.sequences)
        self._data = pd.concat([self._data, filtered_seqs])
        self._data = self._data.drop_duplicates(subset=['sequence'], ignore_index=True)
        added_num_seqs = len(self._data) - prev_num_seqs
        self._log.info(f"Added {added_num_seqs} sequences.")
