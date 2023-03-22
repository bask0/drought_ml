
import sys
import os
from argparse import ArgumentParser

from preprocessing.cube_harmonize_utils import create_dummy, write_data, add_anomalies, merge_stats


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
        required=('--create' not in sys.argv) and ('--anomalies' not in sys.argv) and ('--seasonality' not in sys.argv) and ('--merge_stats' not in sys.argv),
        help='dataset to add to the cube. Only required if none `--create`, `--anomalies`, and `--merge_sats` are passed.')
    parser.add_argument(
        '--anomalies',
        type=str,
        help='if passed, anomalies are calculated for the given variable. One of `lst`, `fvc`.')
    parser.add_argument(
        '--seasonality',
        type=str,
        help='if passed, seasonality is calculated for the given variable. One of `t2m`, `tp`, `ssrd`, `rH_cf`.')
    parser.add_argument(
        '--merge_stats',
        action='store_true',
        help='if passed, the standard deviation and mean are calculated from the `<var>_stats` variables.')
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
        if args.anomalies is None and args.seasonality is None and args.merge_stats is False:
            write_data(in_path=args.in_file, out_path=args.out_file, dryrun=args.dryrun)
        elif args.anomalies is not None:
            if args.anomalies not in ('lst', 'fvc'):
                raise ValueError(
                    f'argument `anomalies` must be one of (\'fvc\', \'lst\'), is {args.anomalies}.'
                )
            add_anomalies(var=args.anomalies, out_path=args.out_file, dryrun=args.dryrun, num_proc=12)
        elif args.seasonality is not None:
            if args.seasonality not in ('t2m', 'tp', 'ssrd', 'rh_cf'):
                raise ValueError(
                    f'argument `anomalies` must be one of (\'t2m\', \'tp\', \'ssrd\', \'rh_cf\'), is {args.seasonality}.'
                )
            add_anomalies(var=args.seasonality, out_path=args.out_file, dryrun=args.dryrun, num_proc=12, msc_only=True)
        else:
            merge_stats(out_path=args.out_file)
