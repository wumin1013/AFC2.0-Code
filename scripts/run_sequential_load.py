"""Thin explicit CLI: no web scheduler and no implicit data release."""
import argparse
import json
from pathlib import Path
import sys
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spindle_power.load_prediction.sequential_service import SequentialService

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('action', choices=('score','release','train','memory_sensitivity'))
    parser.add_argument('--group', choices=('O1','O2','O3'))
    parser.add_argument('--released-groups', nargs='+')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from spindle_power.load_prediction.load_device import configure_load_device
    configure_load_device('cpu', cpu_threads=4, cpu_interop_threads=1)
    service = SequentialService(args.case, args.output)
    if args.action == 'train':
        if args.group or not args.released_groups:
            parser.error('train需要显式--released-groups')
        result = service.train(args.released_groups, resume=args.resume)
    else:
        if not args.group or args.resume or args.released_groups:
            parser.error('score/release只接受--group')
        result = getattr(service, args.action)(args.group)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__': main()
