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
    filelist = ['258', '241', '252', '324', '614', '033', '045', '027', '585', '107', '192', '989', '210', '607', '009', '289', '469', '645', '611', '951', '356', '198', '071', '943', '827', '863', '919', '712', '292', '321', '819', '178', '598', '642', '542', '628', '262', '109', '981', '246', '088', '883', '644', '479', '615', '554', '876', '438', '588', '284', '683', '965', '106', '097', '609', '953', '866', '221', '060', '272', '183', '222', '994', '153', '520', '035', '986', '633', '206', '253', '990', '294', '987', '337', '263', '420', '228', '716', '264', '904', '640', '882', '288', '817', '670', '004', '470', '266', '375', '055', '054', '657', '982', '719', '522', '985', '456', '437', '396', '066', '948', '301', '999', '251', '261', '397', '649', '668', '602', '942', '044', '924', '980', '441', '741', '920', '786', '853', '013', '168', '679', '599', '015', '200', '889', '630', '565', '661', '032', '720', '189', '150', '134', '616', '448', '834', '760', '852', '159', '128', '175', '271', '966', '944', '654', '988', '828', '254', '589', '687', '974', '468', '969', '399', '731', '671', '101', '664', '452', '046', '376', '381', '036', '677', '993', '897', '992', '771', '062', '439', '947', '706', '638', '550', '892', '917', '360', '952', '896', '651', '556', '635', '835', '878', '688', '737', '339', '779', '811', '648', '568', '112', '529', '361', '481', '147', '938', '794', '891', '830', '672', '960', '435', '096', '945', '849', '623', '434', '008', '257', '596', '665', '816', '392', '572', '488']

    print(f"Manually downloading {len(filelist)} specific original videos...")

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

