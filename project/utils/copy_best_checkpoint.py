
from argparse import ArgumentParser
import torch
from glob import glob
import os
import math
import shutil

def find_best_checkpoint(base_dir: str) -> str:
    tune_path = os.path.join(base_dir, 'tune', 'trial*', 'checkpoints', 'best.ckpt')

    checkpoints = glob(tune_path)
    if len(checkpoints) == 0:
        raise RuntimeError(
            f'no ckeckpoints found with pattern `{tune_path}`. Did you run hyper parameter tuning? '
            'Alternatively, run `run_cv.sh` with default config by passing option `-d`.'
        )

    current_best_score = math.inf
    best_checkpoint = None

    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location=torch.device('cpu'))

        es = [value for key, value in state['callbacks'].items() if 'EarlyStopping' in key]
        if len(es) == 0:
            raise RuntimeError(
                f'Cannot infer best score from checkpoint `{checkpoint}`; no EarlyStopping state dict found. '
            )

        if 'best_score' not in es[0]:
            raise RuntimeError(
                f'Cannot infer best score from checkpoint `{checkpoint}`; key \'best_score\' not found in EerlyStopping state.'
            )

        best_score = es[0]['best_score'].item()
        if best_score < current_best_score:
            current_best_score = best_score
            best_checkpoint = checkpoint

    if best_checkpoint is None:
        raise RuntimeError(
            'could not find best checkpoint.'
        )

    return best_checkpoint


def copy_best_config(base_dir: str) -> str:
    best_checkpoint = find_best_checkpoint(base_dir)
    trial_path = best_checkpoint.split('checkpoint')[0]
    best_config_file = os.path.join(trial_path, 'config.yaml')

    shutil.copy2(best_config_file, os.path.join(base_dir, 'cv', 'best_config.yaml'))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        'path',
        type=str,
        help='Path to an experiment containing `tune` directory.'
    )
    args = parser.parse_args()

    copy_best_config(args.path)
