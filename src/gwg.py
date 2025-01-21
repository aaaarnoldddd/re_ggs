from typing import List, Optional, Tuple
import logging
import hydra
import pytorch_lightning as L
import pyrootutils
import copy
import time
import os
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import pandas as pd
import torch
# from src.data.sequence_dataset import PreScoredSequenceDataset
import pickle as pkl

pyrootutils.setup_root(__file__, indicator="environment.yaml", pythonpath=True)
from src.model.gwg_module import GwgPairSampler
from src.data import my_dataset

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def _worker_fn(args):
    worker_i, exp_cfg, inputs = args
    model = GwgPairSampler(
        **exp_cfg,
        device=f"cuda:0"
    )
    all_candidates, all_acceptance_rates = [], []
    for batch in inputs:
        candidates, acceptance_rate = model(batch)
        all_candidates.append(candidates)
        all_acceptance_rates.append(acceptance_rate)
    log.info(f'Done with worker: {worker_i}')
    return all_candidates, all_acceptance_rates

def generate_pairs(cfg, sample_write_path):
    run_cfg = cfg.run
    L.seed_everything(cfg.run.seed)
    dataset = my_dataset(**cfg.data)
    # print("hi")
    exp_cfg = dict(cfg.experiment)
    epoch = 0
    start_time = time.time()
    acceptance_rates = []
    while epoch < run_cfg.max_epochs and len(dataset):
        epoch += 1
        epoch_start_time = time.time()
        batch_size = len(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Run sampling with workers
        batches_per_worker = [[]]
        for i, batch in enumerate(dataloader):
            batches_per_worker[i].append(batch)
            if run_cfg.debug:
                break
        log.info(f"using GPU: {torch.device('cuda')}" )
        all_worker_outputs = [
            _worker_fn((0, exp_cfg, batches_per_worker[0]))
        ]

        # Process results.
        epoch_pair_count = 0
        candidate_seqs = []

        for worker_results, acceptance_rate in all_worker_outputs:
            for new_pairs in worker_results:
                if new_pairs is None:
                    continue
            
                candidate_seqs.append(
                    new_pairs[['mutant_sequence', 'mutant_score']].rename(
                        columns={'mutant_sequence': 'sequence', 'mutant_score': 'score'}
                    )
                )
            acceptance_rates.append(acceptance_rate[0])
            epoch_pair_count += dataset.add_pairs(new_pairs, epoch)
        if len(candidate_seqs) > 0:
            candidate_seqs = pd.concat(candidate_seqs)
            candidate_seqs.drop_duplicates(subset='sequence', inplace=True)
        epoch_elapsed_time = time.time() - epoch_start_time

        log.info(f"Epoch {epoch} finished in {epoch_elapsed_time:.2f} seconds")
        log.info("------------------------------------")
        log.info(f"Generated {epoch_pair_count} pairs in this epoch")
        log.info(f"Acceptance rate = {acceptance_rate[0]:.2f}")
        dataset.reset()
        if epoch < run_cfg.max_epochs and len(candidate_seqs) > 1:
            dataset.add(candidate_seqs)
            if cfg.data.clustering:
                dataset.cluster()
        log.info(f"Next dataset = {len(dataset)} sequences")
    dataset.pairs.to_csv(sample_write_path, index=False)
    # save acceptance_rates to pkl
    cluster_centers_path = sample_write_path.replace('.csv', '_cluster_centers.csv')
    dataset._cluster_centers.to_csv(cluster_centers_path, index=False)
    acceptance_rates_path = sample_write_path.replace('.csv', '_acceptance_rates.pkl')
    with open(acceptance_rates_path, 'wb') as f:
        pkl.dump(acceptance_rates, f)
        
    elapsed_time = time.time() - start_time
    log.info(f'Finished generation in {elapsed_time:.2f} seconds.')
    log.info(f'Samples written to {sample_write_path}.')




@hydra.main(version_base=None , config_path="../config/" , config_name="gwg.yaml")
def main(cfg):
    cfg_run = cfg.run
    cfg_data = cfg.data
    predictor_dir = cfg.experiment.predictor_dir
    s = predictor_dir.split('/')
    s_copy = s
    for i,seg in enumerate(s):
        if seg == 'ckpt':
            s[i] = 'data'
        if 'percentile' in seg:
            pos = i
            break
    base_data_dir = '/'.join(s[:i]) + '_' + s[i]
    base_data_path = os.path.join(
        base_data_dir,
        "filtered_dataset.csv"
    )

    if os.path.exists(base_data_path):
        # log.info(f"Raw data is saved to {base_data_path}")
        pass
    else:
        ValueError("path not exist")

    cfg.data.csv_path = base_data_path

    sample_write_dir = os.path.join(
        predictor_dir,
        cfg_run.run_name
    )
    os.makedirs(sample_write_dir, exist_ok=True)

    if os.path.exists(base_data_path):
        print("path exists")

    con_write_path = os.path.join(
        sample_write_dir,
        "config.yaml"
    )
    with open(con_write_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    
    sample_write_path = os.path.join(
        sample_write_dir,
        f"seed_{cfg_run.seed}.csv"
    )

    generate_pairs(cfg, sample_write_path)



if __name__ == "__main__":
    main()