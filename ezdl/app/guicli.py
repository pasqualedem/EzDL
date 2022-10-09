import argparse
from ezdl.app.gui import streamlit_entry

parser = argparse.ArgumentParser(description='EzDL App')

parser.add_argument('--resume', required=False, action='store_true',
                    help='Resume the experiment', default=False)
parser.add_argument('-d', '--dir', required=False, type=str,
                    help='Set the local tracking directory', default=None)
parser.add_argument('-f', '--file', required=False, type=str,
                    help='Set the config file', default=None)
parser.add_argument('--share', required=False, action='store_true',
                    help='Tells if share the app', default=False)
parser.add_argument("--grid", type=int, help="Select the first grid to start from", default=None)
parser.add_argument("--run", type=int, help="Select the run in grid to start from", default=None)


def cli():
    args = parser.parse_args()
    exp_settings = dict(
        start_from_grid=args.grid,
        start_from_run=args.run,
        resume=args.resume,
        tracking_dir=args.dir
    )
    streamlit_entry(args.file, exp_settings, args.share)


if __name__ == '__main__':
    cli()
