from __future__ import print_function
import os
import pickle

import numpy
from data import get_test_loader
import re
import numpy as np
from vocab import Vocabulary  # NOQA
import torch
from model import VSE
from glob import glob

from tqdm.autonotebook import tqdm

from evaluation import encode_data

import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

baseline_files = glob('caps_files/test/*beam*_reformat.json') + glob('caps_files/test/*greedy*_reformat.json')
files_2d = glob('caps_files/test/*d-2*_reformat.json')
files_4d = glob('caps_files/test/*d-4*_reformat.json')
files_9d = glob('caps_files/test/*d-9*_reformat.json')

cluster_file_2d = 'caps_files/test/image_clusters/image_clusters_test_3.pkl'
cluster_file_4d = 'caps_files/test/image_clusters/image_clusters_test_5.pkl'
cluster_file_9d = 'caps_files/test/image_clusters/image_clusters_test_10.pkl'

print('baseline:', baseline_files)
print('2d files:', files_2d)
print('4d files:', files_4d)
print('9d files:', files_9d)
print('\n')

print('cuda:', torch.cuda.is_available())

# model_path="data/runs/coco_vse++/model_best.pth.tar"
model_path = "data/runs/coco_vse++_resnet_restval/model_best.pth.tar"
image_path = "/home/vu48pok/.data/compling/data/corpora/external/MSCOCO/COCO/val2014/"
data_path = "data/"
vocab_path = "vocab/"
split = "test"
on_gpu = True

device = 'cpu' if not on_gpu else 'cuda'
checkpoint = torch.load(model_path, map_location=torch.device(device))
opt = checkpoint['opt']

if data_path is not None:
    opt.data_path = data_path

if vocab_path is not None:
    opt.vocab_path = vocab_path

# load vocabulary used by the model
with open(os.path.join(opt.vocab_path,
                       '%s_vocab.pkl' % opt.data_name), 'rb') as f:
    vocab = pickle.load(f)
opt.vocab_size = len(vocab)

# construct model
model = VSE(opt)

# load model state
model.load_state_dict(checkpoint['model'])


def evaluate_file(
        caption_file, cluster_file, model=model,
        vocab=vocab, image_path=image_path, opt=opt):

    # read cluster file
    with open(cluster_file, 'rb') as f:
        clusters = pickle.load(f)
        n_dist = len(clusters[0])

    # read caption file to make sure
    # the image ids align with the cluster targets
    with open(caption_file, 'r') as f:
        data = json.load(f)
        img_ids = [i['image_id'] for i in data['annotations']]
        assert [i[0] for i in clusters] == img_ids

    # initialize dataloader with captions from caption_file
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt,
                                  image_location=image_path,
                                  caption_file=caption_file
                                  )

    print('Computing results...')
    # embed images and captions in a joint space
    img_embs, cap_embs = encode_data(model, data_loader, on_gpu=on_gpu)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0], cap_embs.shape[0]))

    # create df containing ordered image ids
    ann_df = pd.DataFrame(data_loader.dataset.coco[0].anns).T.set_index('id')

    # create dicts to align index positions and image ids
    idx2imgid = ann_df['image_id'].to_dict()
    imgid2idx = {v: k for k, v in idx2imgid.items()}

    # initialize empty array with shape [5000, n_dist]
    # (where n_dist in [2, 4, 9])
    all_ranks = np.zeros((len(cap_embs), n_dist))

    # iterate through range from 0-4999
    for i in range(len(cap_embs)):
        # get caption embedding for current index
        cemb = cap_embs[i]
        # get target image id for current index
        target = ann_df.loc[i].image_id
        # get image cluster where target is the first image
        cluster = [j for j in clusters if j[0] == target][0]
        assert target == cluster[0]

        # get indices from cluster image ids
        idx_array = np.array([imgid2idx[j] for j in cluster])
        # get image embeddings for cluster entries
        iemb = img_embs[idx_array]

        # get cosine similarity between current caption embedding
        # and all image embeddings
        cosines = np.zeros((len(iemb), 1))
        for j in range(len(iemb)):
            cosines[j] = cosine_similarity(
                cemb.reshape(1, -1), iemb[j].reshape(1, -1))
        # get ranks by sorting the cosines (in descending order)
        ranks = np.argsort(cosines.ravel())[::-1]

        # dots = cemb @ iemb.T
        # sort indices according to dot product
        # ranks = np.argsort(dots.ravel())[::-1]
        # add to all_ranks array
        all_ranks[i] = ranks

    # determine how often the target image is ranked first
    target_positions = np.where(all_ranks == 0)[1]
    # compute accuracy
    acc = len(target_positions[target_positions == 0]) / len(target_positions)

    return acc, all_ranks, target_positions


def get_lambda(s):
    match = re.search(r'_l-(\d-\d)', s)
    if match:
        d = match.group(1).replace('-', '.')
        return float(d)
    else:
        return None


def get_rationality(s):
    match = re.search(r'_r-(\d-\d)', s)
    if match:
        d = match.group(1).replace('-', '.')
        return float(d)
    else:
        return None


def get_n_distractors(s):
    match = re.search(r'_d-(\d)', s)
    if match:
        d = match.group(1)
        return int(d)
    else:
        return None


def get_topk(s):
    match = re.search(r'_k-(\d+)', s)
    if match:
        d = match.group(1)
        return int(d)
    else:
        return None


def get_topp(s):
    match = re.search(r'_p-(\d-\d)', s)
    if match:
        d = match.group(1).replace('-', '.')
        return float(d)
    else:
        return None


def get_temperature(s):
    match = re.search(r'_t-(\d-\d)', s)
    if match:
        d = match.group(1).replace('-', '.')
        return float(d)
    else:
        return None


def get_segmentation(s):
    if s.endswith('char'):
        return 'char'
    else:
        return 'word'


def get_method(s):
    match = re.search(r'((val)|(test))_([a-z\_]+)_d\-', s)
    if match:
        return (match.group(4))
    else:
        return None


cluster_file = cluster_file_2d
results_2d = dict()
for file in tqdm(baseline_files + files_2d):
    key = os.path.split(file)[-1].replace('_reformat.json', '')
    print(key)

    acc, all_ranks, target_positions = evaluate_file(
        file, cluster_file, model, vocab)

    results_2d[key] = {
            'acc': acc,
            'all_ranks': all_ranks,
            'target_positions': target_positions
        }

cluster_file = cluster_file_4d
results_4d = dict()
for file in tqdm(baseline_files + files_4d):
    key = os.path.split(file)[-1].replace('_reformat.json', '')
    print(key)

    acc, all_ranks, target_positions = evaluate_file(
        file, cluster_file, model, vocab)

    results_4d[key] = {
            'acc': acc,
            'all_ranks': all_ranks,
            'target_positions': target_positions
        }

cluster_file = cluster_file_9d
results_9d = dict()
for file in tqdm(baseline_files + files_9d):
    key = os.path.split(file)[-1].replace('_reformat.json', '')
    print(key)

    acc, all_ranks, target_positions = evaluate_file(
        file, cluster_file, model, vocab)

    results_9d[key] = {
            'acc': acc,
            'all_ranks': all_ranks,
            'target_positions': target_positions
        }

accs_2d = pd.DataFrame(results_2d).T[['acc']]
accs_4d = pd.DataFrame(results_4d).T[['acc']]
accs_9d = pd.DataFrame(results_9d).T[['acc']]

accs_2d['_lambda'] = accs_2d.index.map(get_lambda)
accs_2d['rationality'] = accs_2d.index.map(get_rationality)
accs_2d['method'] = accs_2d.index.map(get_method)
accs_2d['segmentation'] = accs_2d.index.map(get_segmentation)
accs_2d['n_distractors'] = 2
accs_2d['temperature'] = accs_2d.index.map(get_temperature)
accs_2d['topp'] = accs_2d.index.map(get_topp)
accs_2d['topk'] = accs_2d.index.map(get_topk)

accs_4d['_lambda'] = accs_4d.index.map(get_lambda)
accs_4d['rationality'] = accs_4d.index.map(get_rationality)
accs_4d['method'] = accs_4d.index.map(get_method)
accs_4d['segmentation'] = accs_4d.index.map(get_segmentation)
accs_4d['n_distractors'] = 4
accs_4d['temperature'] = accs_4d.index.map(get_temperature)
accs_4d['topp'] = accs_4d.index.map(get_topp)
accs_4d['topk'] = accs_4d.index.map(get_topk)

accs_9d['_lambda'] = accs_9d.index.map(get_lambda)
accs_9d['rationality'] = accs_9d.index.map(get_rationality)
accs_9d['method'] = accs_9d.index.map(get_method)
accs_9d['segmentation'] = accs_9d.index.map(get_segmentation)
accs_9d['n_distractors'] = 9
accs_9d['temperature'] = accs_9d.index.map(get_temperature)
accs_9d['topp'] = accs_9d.index.map(get_topp)
accs_9d['topk'] = accs_9d.index.map(get_topk)

pd.concat(
        [accs_2d, accs_4d, accs_9d]
    ).to_csv(
        'vsepp_accs_test.csv'
    )
