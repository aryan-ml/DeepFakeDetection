#!/usr/bin/env python
""" Downloads FaceForensics++ (v2) sample data
Example:
    python download.py -d Deepfakes -c c23 -t videos --num_videos 5 ./data/raw
"""
import argparse
import os
import urllib.request
import tempfile
import json
import sys
import time
from tqdm import tqdm
from os.path import join

# Constants
DATASETS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
}
COMPRESSION = ['raw', 'c23', 'c40']
SERVERS = {
    'EU': 'http://canis.vc.in.tum.de:8100/',
    'EU2': 'http://kaldir.vc.in.tum.de/faceforensics/',
    'CA': 'http://falas.cmpt.sfu.ca:8100/',
}

def download_file(url, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    if not os.path.exists(out_file):
        tmp, tmp_name = tempfile.mkstemp()
        os.close(tmp)
        with tqdm(unit='B', unit_scale=True, desc=os.path.basename(out_file)) as pbar:
            urllib.request.urlretrieve(url, tmp_name,
                                       reporthook=lambda b, bs, ts: pbar.update(bs))
        os.rename(tmp_name, out_file)
    else:
        print(f"Skipping existing file {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='Deepfakes', choices=list(DATASETS.keys()))
    parser.add_argument('-c', '--compression', default='c23', choices=COMPRESSION)
    parser.add_argument('-t', '--type', default='videos', choices=['videos'])
    parser.add_argument('--num_videos', type=int, default=5)
    parser.add_argument('--server', default='EU', choices=list(SERVERS.keys()))
    parser.add_argument('output_path', help='Output directory path')
    args = parser.parse_args()

    base_url = SERVERS[args.server] + 'v3/'
    dataset_path = DATASETS[args.dataset]
    filelist_url = base_url + 'misc/filelist.json'
    print("Fetching file list...")
    file_pairs = json.loads(urllib.request.urlopen(filelist_url).read().decode("utf-8"))
    filelist = ['_'.join(pair) for pair in file_pairs][:args.num_videos]

    dataset_url = base_url + f"{dataset_path}/{args.compression}/{args.type}/"
    out_dir = join(args.output_path, dataset_path, args.compression, args.type)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading {len(filelist)} videos from {dataset_url}")

    for name in filelist:
        file_url = dataset_url + name + '.mp4'
        out_file = join(out_dir, name + '.mp4')
        download_file(file_url, out_file)

if __name__ == "__main__":
    main()

