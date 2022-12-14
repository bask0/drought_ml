
import sys
import os
from argparse import ArgumentParser

from preprocessing.cube_harmonize_utils import create_dummy, write_data, add_anomalies


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
        required=('--create' not in sys.argv) and ('--anomalies' not in sys.argv),
        help='dataset to add to the cube. Only required if not `--create`')
    parser.add_argument(
        '--anomalies',
        type=str,
        help='if passed, anomalies are calculated for the given variable. One of `lst`, `fvc`.')
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
        if args.anomalies is None:
            print(f'   > Writing file {args.in_file}')
            write_data(in_path=args.in_file, out_path=args.out_file, dryrun=args.dryrun)
            print('   > Done')
        else:
            if args.anomalies not in ('lst', 'fvc'):
                raise ValueError(
                    f'argument `anomalies` must be one of (\'fvc\', \'lst\'), is {args.anomalies}.'
                )
            add_anomalies(var=args.anomalies, out_path=args.out_file, dryrun=args.dryrun, num_proc=12)
            print('   > Done')
