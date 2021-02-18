from __future__ import print_function
import os
from os.path import join
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
import argparse


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)


def get_n_distractors(s):
    match = re.search(r'_d-(\d)', s)
    if match:
        d = match.group(1)
        return int(d)
    else:
        return None


def evaluate_file(caption_file, cluster_file,
                  model, vocab, image_path, opt, split
                  ):

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
    opt.data_path = args.data_path
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt,
                                  image_location=image_path,
                                  caption_file=caption_file
                                  )

    print('Computing results...')
    # embed images and captions in a joint space
    on_gpu = True if device == 'cuda' else False
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
    img_ids = np.zeros((len(cap_embs), n_dist))

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

        # add to all_ranks array
        all_ranks[i] = ranks
        img_ids[i] = cluster

    # determine how often the target image is ranked first
    target_positions = np.where(all_ranks == 0)[1]
    # compute accuracy
    acc = len(target_positions[target_positions == 0]) / len(target_positions)

    return acc, all_ranks, target_positions, img_ids


def main(args):

    checkpoint = torch.load(args.model_path, map_location=torch.device(device))
    opt = checkpoint['opt']

    # load vocabulary used by the model
    with open(join(args.vocab_path,
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)

    # construct model
    model = VSE(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    # sort input files according to number of get_n_distractors
    files_0d = glob(join(args.input_dir, '*d-na*_reformat.json')) #  baseline / diversity
    files_2d = glob(join(args.input_dir, '*d-2*_reformat.json'))
    files_4d = glob(join(args.input_dir, '*d-4*_reformat.json'))
    files_9d = glob(join(args.input_dir, '*d-9*_reformat.json'))

    results = []

    for files, n_dist in zip(
                [files_2d, files_4d, files_9d],
                [2, 4, 9]
            ):

        # skip if the current list is empty
        if len(files) == 0:
            continue

        # set template for clusters

        if args.random_distractors:
            cluster_template = 'random_image_clusters_{split}_{n}.pkl'
        else:
            cluster_template = 'image_clusters_{split}_{n}.pkl'

        # select corresponding cluster file
        cluster_file = join(
            args.cluster_dir,
            cluster_template.format(
                split=args.split, n=str(n_dist + 1))
            )

        res = dict()
        t = tqdm(files + files_0d)

        for file in t:
            # get file name
            key = os.path.split(file)[-1].replace('_reformat.json', '')
            t.set_description(key)
            t.refresh()

            # evaluate file
            acc, all_ranks, target_positions, cluster = evaluate_file(
                    file, cluster_file, model=model,
                    vocab=vocab, image_path=args.image_path, opt=opt, split=args.split
                    )

            # store results for current file
            res[key] = {
                    'acc': acc,
                    'all_ranks': all_ranks.astype(int).tolist(),
                    'target_positions': target_positions.astype(int).tolist(),
                    'n_eval_dists': n_dist,
                    'image_cluster': cluster.astype(int).tolist()
                }

        # store results for current set of files
        results.append(res)

    # transform results into dataframe
    res_dfs = [pd.DataFrame(r).T[['n_eval_dists', 'acc', 'all_ranks', 'target_positions', 'image_cluster']] for r in results]
    # merge results
    concat_df = pd.concat(res_dfs)
    # write to file
    concat_df.to_csv(args.out_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        help='the directory of the files to be processed')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--cluster_dir', type=str,
                        default='')
    parser.add_argument('--model_path', type=str,
                        default='data/runs/coco_vse++_resnet_restval/model_best.pth.tar',
                        help='VSE++ model path')
    parser.add_argument('--image_path', type=str,
                        default='/home/vu48pok/.data/compling/data/corpora/external/MSCOCO/COCO/val2014/',
                        help='path to COCO images')
    parser.add_argument('--data_path', type=str,
                        default='./data/',
                        help='VSE++ data path')
    parser.add_argument('--vocab_path', type=str,
                        default='./vocab/',
                        help='VSE++ vocab path')
    parser.add_argument('--out_file', type=str,
                        default='./discriminativeness_results.csv',
                        help='path for the output csv file')
    parser.add_argument('--random_distractors', action='store_true')
    args = parser.parse_args()

    if len(args.cluster_dir) == 0:
        args.cluster_dir = join(args.input_dir, 'image_clusters')
    if args.random_distractors:
        print('using random clusters')

    main(args)
