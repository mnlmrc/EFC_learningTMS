import PcmPy as pcm
import numpy as np
import os
import globals as gl
import argparse
import pandas as pd
from imaging_pipelines.util import extract_mle_corr
from util import prew_res_from_part_mean
import time

chord_mapping = {
    'C0': 0,
    'C1': 1,
    'C2': 2,
    'C3': 3,
    'C4': 4,
}

def mle_correlation(file_list, x, y, x_pre=False, y_pre=False):
    Y, G = [], []
    Mflex = pcm.CorrelationModel("flex", num_items=4, corr=None, cond_effect=True)
    for file in file_list:

        df = pd.read_csv(file, sep='\t')
        #df.Chord = df.Chord.map(chord_mapping)
        df_x = df[df['condition']==x]
        df_y = df[df['condition']==y]

        x_raw = df_x[gl.channels].to_numpy() if x_pre is False else df_x[gl.channels_PRE].to_numpy()
        y_raw = df_y[gl.channels].to_numpy() if y_pre is False else df_y[gl.channels_PRE].to_numpy()

        x_g, Z, part_vec = pcm.group_by_condition(x_raw, df_x.Chord, df_x.BN, axis=0)
        y_g, _, _ = pcm.group_by_condition(y_raw, df_y.Chord, df_y.BN, axis=0)

        cond_vec_x = np.argmax(Z, axis=1)
        cond_vec_y = cond_vec_x + 4

        obs_des = {'cond_vec': np.r_[cond_vec_x, cond_vec_y],
                   'part_vec': np.r_[part_vec, part_vec]}

        data = np.r_[x_g, y_g]
        data = data - data.mean(axis=1, keepdims=True)
        # data_prewhitened = prew_res_from_part_mean(data, obs_des['part_vec'])
        Y.append(pcm.dataset.Dataset(data, obs_descriptors=obs_des))
        G.append(pcm.est_G_crossval(
            Y[-1].measurements,
            Y[-1].obs_descriptors['cond_vec'],
            Y[-1].obs_descriptors['part_vec'])[0])

    _, theta = pcm.fit_model_individ(Y, Mflex, fixed_effect='block', fit_scale=False, verbose=False)
    _, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect='block', fit_scale=True, verbose=False)

    r_indiv, r_group, SNR = extract_mle_corr(Mflex, theta[0], theta_gr[0])

    return r_indiv, r_group, SNR, np.array(G)

def mle_correlation3(file_list):

    Y, G = [], []
    Mflex = pcm.CorrelationModel("flex", num_items=4, corr=None, cond_effect=True)
    for file in file_list:

        df = pd.read_csv(file, sep='\t')
        df.Chord = df.Chord.map(chord_mapping)
        df_pre = df[df['condition']=='TMS']
        df_tms = df[df['condition']=='TMS']
        df_vol = df[df['condition']=='Chord']

        pre = df_pre[gl.channels_PRE].to_numpy()
        tms = df_tms[gl.channels].to_numpy()
        vol = df_vol[gl.channels].to_numpy()

        pre_g, Z, part_vec = pcm.group_by_condition(pre, df_pre.Chord, df_pre.BN, axis=0)
        tms_g, _, _ = pcm.group_by_condition(tms, df_tms.Chord, df_tms.BN, axis=0)
        vol_g, _, _ = pcm.group_by_condition(vol, df_vol.Chord, df_vol.BN, axis=0)

        cond_vec_pre = np.argmax(Z, axis=1)
        cond_vec_tms = cond_vec_pre + 4
        cond_vec_vol = cond_vec_pre + 8

        obs_des = {'cond_vec': np.r_[cond_vec_pre, cond_vec_tms, cond_vec_vol],
                   'part_vec': np.r_[part_vec, part_vec,part_vec]}

        data = np.r_[pre_g, tms_g, vol_g]
        data = data - data.mean(axis=1, keepdims=True)
        data_prewhitened = prew_res_from_part_mean(data, obs_des['part_vec'])
        Y.append(pcm.dataset.Dataset(data_prewhitened, obs_descriptors=obs_des))
        G.append(pcm.est_G_crossval(
            Y[-1].measurements,
            Y[-1].obs_descriptors['cond_vec'],
            Y[-1].obs_descriptors['part_vec'])[0])

    return np.array(G)


def main(args):
    file_list = []
    for sn in args.sns:
        file_list.append(os.path.join(gl.baseDir, gl.behavDir, f'p{sn}_testing', 'single_trial.tsv'))
    if args.what == 'corr_3':
        G = mle_correlation3(file_list)
        save_path = os.path.join(os.path.join(gl.baseDir, gl.pcmDir))
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'G_obs.pre-tms-vol.npy'), G)
    if args.what=='corr_tms-vol':
        r_indiv, r_group, SNR, G = mle_correlation(file_list, 'TMS', 'Chord')
        df = pd.DataFrame()
        df['participant_id'] = args.sns
        df['r_indiv'] = r_indiv
        df['r_group'] = r_group
        df['SNR'] = SNR
        save_path = os.path.join(os.path.join(gl.baseDir, gl.pcmDir))
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(os.path.join(save_path, 'corr_tms-vol.tsv'), sep='\t', index=False)
        np.save(os.path.join(save_path, 'G_obs.tms-vol.npy'), G)
    if args.what=='corr_pre-vol':
        r_indiv, r_group, SNR, G = mle_correlation(file_list, 'TMS', 'Chord', x_pre=True)
        df = pd.DataFrame()
        df['participant_id'] = args.sns
        df['r_indiv'] = r_indiv
        df['r_group'] = r_group
        df['SNR'] = SNR
        save_path = os.path.join(os.path.join(gl.baseDir, gl.pcmDir))
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(os.path.join(save_path, 'corr_pre-vol.tsv'), sep='\t', index=False)
        np.save(os.path.join(save_path, 'G_obs.pre-vol.npy'), G)
    if args.what=='corr_pre-tms':
        r_indiv, r_group, SNR, G = mle_correlation(file_list, 'TMS', 'TMS', y_pre=True)
        df = pd.DataFrame()
        df['participant_id'] = args.sns
        df['r_indiv'] = r_indiv
        df['r_group'] = r_group
        df['SNR'] = SNR
        save_path = os.path.join(os.path.join(gl.baseDir, gl.pcmDir))
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(os.path.join(save_path, 'corr_pre-tms.tsv'), sep='\t', index=False)
        np.save(os.path.join(save_path, 'G_obs.pre-tms.npy'), G)

if __name__ == '__main__':

    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--sns', nargs='+', default=[100, 102, 103, 104], type=int)

    args = parser.parse_args()

    main(args)

    end = time.time()
    print(f'Time elapsed: {end - start} seconds')