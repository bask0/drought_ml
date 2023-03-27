
import sys
import os
from argparse import ArgumentParser

from preprocessing.cube_harmonize_utils import create_dummy, write_data


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        '--create',
        action='store_true',
        help='initialize dataset')
    parser.add_argument(
        '--start_year',
        type=int,
        default=2002,
        help='start year for dataet initialization')
    parser.add_argument(
        '--end_year',
        type=int,
        default=2021,
        help='end year for dataet initialization')
    parser.add_argument(
        '-o',
        '--out_file',
        type=str,
        default='./test.zarr',
        help='dataset path (end with .zarr)')
    parser.add_argument(
        '-f',
        '--in_file',
        type=str,
        required=('--create' not in sys.argv),
        help='dataset to add to the cube. Only required if none `--create`, `--anomalies`, and `--merge_sats` are passed.')
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='do dry run')

    args = parser.parse_args()

    if args.create:
        print('   > Initializing cube')
        create_dummy(start_year=args.start_year, end_year=args.end_year, out_path=args.out_file)
        print('   > Done')
    else:
        if not os.path.isdir(args.out_file):
            raise FileNotFoundError(
                f'`out_path={args.out_file}` is not a directory. '
                'Initialize the dataset first with flag `--create_dataset`.')

        write_data(in_path=args.in_file, out_path=args.out_file, dryrun=args.dryrun)
