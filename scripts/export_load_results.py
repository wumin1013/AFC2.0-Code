"""Explicit Python table/figure entry point for accepted load results."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spindle_power.load_prediction.result_export import export_initial,export_sequential

def main():
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=('initial','sequential'))
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-plot',action='store_true')
    a=p.parse_args()
    fn=export_initial if a.kind=='initial' else export_sequential
    print(json.dumps(fn(a.source,a.output,plot=not a.no_plot),ensure_ascii=False))

if __name__=='__main__': main()
