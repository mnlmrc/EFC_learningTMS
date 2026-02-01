import os
import numpy as np
import pandas as pd
import globals as gl
import time
import PcmPy as pcm
import argparse
from util import prew_res_from_part_mean

chord_mapping = {
    'C0': 0,
    'C1': 1,
    'C2': 2,
    'C3': 3,
    'C4': 4,
}

def main(args):
    if args.what=='G_obs':
        path = os.path.join(gl.baseDir, gl.behavDir, f'p{args.sn}_testing')
        df = pd.read_csv(os.path.join(path, 'single_trial.tsv'), sep='\t')
        df.Chord = df.Chord.map(chord_mapping)
        condition = ['pre-activation', 'TMS', 'Chord']
        G = np.zeros((len(condition), 4, 4))
        for c, cond in enumerate(condition):
            if cond == 'pre-activation':
                df_tmp = df[df['condition'] == 'TMS']
                Y = df_tmp[gl.channels_PRE].to_numpy()
            else:
                df_tmp = df[df['condition'] == cond]
                Y = df_tmp[gl.channels].to_numpy()

            Y_g, cond_vec, part_vec = pcm.group_by_condition(Y, df_tmp.Chord, df_tmp.BN, axis=0)
            Y_prew = prew_res_from_part_mean(Y_g, part_vec)
            G[c], _ = pcm.est_G_crossval(Y_prew, cond_vec, part_vec, )

        out_path = os.path.join(gl.baseDir, gl.rsaDir, f'subj{args.sn}')
        os.makedirs(out_path, exist_ok=True)
        np.save(os.path.join(out_path, 'G_obs.npy'), G)

    if args.what=='pool_G_obs':
        G = []
        for sn in args.sns:
            path = os.path.join(gl.baseDir, gl.rsaDir, f'subj{sn}')
            G_tmp = np.load(os.path.join(path, 'G_obs.npy'))
            G.append(G_tmp)
        G = np.array(G)
        np.save(os.path.join(gl.baseDir, gl.rsaDir, 'G_obs.npy'), G)

if __name__ == '__main__':

    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default='G_obs')
    parser.add_argument('--sn', type=int, default=100)
    parser.add_argument('--sns', nargs='+', default=[100, 102, 103, 104], type=int)

    args = parser.parse_args()

    main(args)

    end = time.time()
    print(f'Time elapsed: {end - start} seconds')

