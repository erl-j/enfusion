from filesplit.split import Split
from filesplit.merge import Merge
import os
import argparse
#100 MB in bytes is 

def split_checkpoint(checkpoint_path, size_limit=100_000_000):
    output_dir = checkpoint_path.replace('.ckpt', '_split')
    os.makedirs(output_dir, exist_ok=True)
    splitter = Split(inputfile=checkpoint_path, outputdir=output_dir)
    splitter.bysize(size = size_limit)

def merge_checkpoint(split_dir):
    output_filepath = split_dir.replace('_split', '.ckpt')
    output_dir = os.path.dirname(output_filepath)
    merger = Merge(inputdir=split_dir, outputdir=output_dir,outputfilename=os.path.basename(output_filepath))
    merger.merge()

if __name__ == '__main__':

    # first argument is merge or split
    # second argument is the path to the checkpoint or split directory

    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='merge or split')
    
    parser.add_argument('path', type=str, help='path to checkpoint or split directory')

    args = parser.parse_args()

    if args.action == 'split':
        split_checkpoint(args.path)

    elif args.action == 'merge':
        merge_checkpoint(args.path)
