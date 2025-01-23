from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import logging
import os
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
import logging
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

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

        new_data = [(self._encode(x),np.float32(y)) for x,y in 
                    tqdm(zip(filtered.sequence, filtered.score), desc="Processing sequences", total=len(filtered),leave=True)
                    if self._get_min_dist(x) >= min_mutant_dist]

        self._dataset=new_data
        self._log.info("Dataset done")
        self._log.info(f"{len(self._dataset)} samples has been filtered out")

        write_dir = os.path.join(self._task_cfg.task_dir, f"mutant_{self._task_cfg.min_mutant_dist}_percentile_{self._task_cfg.filter_percentile[0]}_{self._task_cfg.filter_percentile[1]}")
        os.makedirs(write_dir, exist_ok=True)
        write_path = os.path.join(write_dir, f"filtered_dataset.csv")

        df = pd.DataFrame({
            "sequence": [self._decode(seq) for seq, _ in new_data],
            "score": [score for _, score in new_data]
        })
                
        self._log.info(f"Write the dataset to {write_path}")
        df.to_csv(write_path, index=False)

    def setup_smoothed(self):
        raw_data = pd.read_csv(self._task_cfg.csv_path)

        self._dataset = [(self._encode(x),np.float32(y)) for x,y in zip(raw_data.sequence, raw_data.score)]
        
        self._log.info(f"get data from {self._task_cfg.csv_path}")
        self._log.info(f'Read in {len(self._dataset)} smoothed sequences.')

    def setup(self, stage=None):
        self._log.info("Start preparing dataset")

        if self._smoothing_params == 'unsmoothed':
            self.setup_unsmoothed()
        else:
            self.setup_smoothed()

        # raw_data = pd.read_csv(self._task_cfg.csv_path)

        # raw_nums = raw_data.shape[0]
        # top_quantile = self._task_cfg.top_quantile
        # filter_per = [self._task_cfg.filter_per[0],self._task_cfg.filter_per[1]]
        # min_mutant_dist = self._task_cfg.min_mutant_dist

        # self._tops = raw_data[raw_data.score >= raw_data.score.quantile(top_quantile)]

        # self._log.info(f"The data between {filter_per[0]*100}% and {filter_per[1]*100}% will be filterer out, which is between {raw_data.score.quantile(filter_per[0])} and {raw_data.score.quantile(filter_per[1])}")

        # filtered = raw_data[raw_data.score.between(raw_data.score.quantile(filter_per[0]), raw_data.score.quantile(filter_per[1]))]

        # new_data = [(self._encode(x),np.float32(y)) for x,y in 
        #             tqdm(zip(filtered.sequence, filtered.score), desc="Processing sequences", total=len(filtered),leave=True)
        #             if self._get_min_dist(x) >= min_mutant_dist]

        # self._dataset=new_data
        # self._log.info("Dataset done")
        # self._log.info(f"{len(self._dataset)} samples has been filtered out")

        # write_dir = os.path.join(self._task_cfg.task_dir, f"mutant_{self._task_cfg.min_mutant_dist}_percentile_{self._task_cfg.filter_per[0]}_{self._task_cfg.filter_per[1]}")
        # os.makedirs(write_dir, exist_ok=True)
        # write_path = os.path.join(write_dir, f"filtered_dataset.csv")

        # df = pd.DataFrame({
        #     "sequence": [self._decode(seq) for seq, _ in new_data],
        #     "score": [score for _, score in new_data]
        # })
                
        # self._log.info(f"Write the dataset to {write_path}")
        # df.to_csv(write_path, index=False)

        self._log.info(f"Use weighted sampling")
        targets = [item[1] for item in self._dataset]  
        targets = np.array(targets)
        adjusted_targets = targets - targets.min() + 1
        weights = 1 / adjusted_targets
        self._sampler = WeightedRandomSampler(weights, len(weights))

    def train_dataloader(self):
        return DataLoader(
            self._dataset,
            batch_size = self._batch_size,
            num_workers = self._num_workers,
            pin_memory = self._pin_memory,
            sampler = self._sampler
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
