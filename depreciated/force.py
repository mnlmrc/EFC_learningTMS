import numpy as np
import pandas as pd
import argparse
import time
import globals as gl
import os

def calc_md(X):
    """

    Args:
        X: timepoints x channels data

    Returns:

    """
    N, m = X.shape
    F1 = X[0]
    FN = X[-1] - F1  # Shift the end point

    shifted_matrix = X - F1  # Shift all points

    d = list()

    for t in range(1, N - 1):
        Ft = shifted_matrix[t]

        # Project Ft onto the ideal straight line
        proj = np.dot(Ft, FN) / np.dot(FN, FN) * FN

        # Calculate the Euclidean distance
        d.append(np.linalg.norm(Ft - proj))

    d = np.array(d)
    MD = d.mean()

    return MD, d

def main(args):
    if args.what == 'single_trial_training':
        path = os.path.join(gl.baseDir, gl.behavDir, f'p{args.sn}_training')
        dat = pd.read_csv(os.path.join(path, f'efcTMS_training_{args.sn}_output.csv'))
        ntrial = dat.shape[0]
        out_dict = {
            'participant_id': [],
            'TN': [],
            'BN': [],
            'Chord': [],
            'ChordSet': [],
            'session': [],
            'success': [],
            'MD': [],
            'ET': []
        }
        for tr in range(ntrial):
            trial_data = pd.read_csv(os.path.join(path, 'raw_data', f'p{args.sn}_trial{tr + 1}.csv'))
            force = trial_data.iloc[:, 1:26].to_numpy() / 1000 # divide by 1000 to get force in N
            planTime = dat.loc[tr].Prep_dur
            holdTime = dat.loc[tr].Match_dur
            start_sample = int(np.floor(planTime * gl.sampling_rate))
            end_sample = force.shape[0] - int(np.floor(holdTime * gl.sampling_rate))
            X = force[start_sample:end_sample]
            MD, d = calc_md(X)
            ET = (end_sample - start_sample) / gl.sampling_rate
            success = dat.loc[tr].Success == 'success'
            out_dict['participant_id'].append(args.sn)
            out_dict['TN'].append(tr + 1)
            out_dict['BN'].append(dat.loc[tr].Block)
            out_dict['MD'].append(MD)
            out_dict['ET'].append(ET)
            out_dict['Chord'].append(dat.loc[tr].Chord)
            out_dict['ChordSet'].append('trained')
            out_dict['session'].append('training')
            out_dict['success'].append(success)
        df = pd.DataFrame(out_dict)
        df.to_csv(os.path.join(path, 'single_trial.tsv'), sep='\t', index=False)
    if args.what == 'single_trial_tms':
        path = os.path.join(gl.baseDir, gl.behavDir, f'p{args.sn}_testing')
        dat = pd.read_csv(os.path.join(path, f'efcTMS_testing_{args.sn}_output.csv'))
        ntrial = dat.shape[0]
        out_dict = {
            'participant_id': [],
            'TN': [],
            'BN': [],
            'Chord': [],
            'condition': [],
            'session': [],
            'success': [],
            'MD': [],
            'ET': [],
            'Chord_cluster': [],
             **{ch: [] for ch in gl.channels},
             **{ch + '_PRE': [] for ch in gl.channels}
        }
        force_tms = np.zeros((ntrial, len(gl.channels), 150))
        for tr in range(ntrial):
            print(f'Processing trial {tr + 1}...')
            trial_data = pd.read_csv(os.path.join(path, 'raw_data', f'p{args.sn}_trial{tr + 1}.csv'))
            force = trial_data[gl.channels].to_numpy() / 1000 # divide by 1000 to get force in N
            planTime = dat.loc[tr].Prep_dur
            holdTime = dat.loc[tr].Match_dur
            Type = dat.loc[tr].Type
            start_sample = int(np.floor(planTime * gl.sampling_rate))
            pre = force[start_sample - int(.1 * gl.sampling_rate):start_sample].mean(axis=0)
            Chord = dat.loc[tr].Chord
            if (Chord == 'C1') | (Chord == 'C3'):
                Chord_cluster = 'D'
            elif (Chord == 'C2') | (Chord == 'C4'):
                Chord_cluster = 'R'
            else:
                Chord_cluster = 'Baseline'
            if Type=='Chord':
                end_sample = force.shape[0] - int(np.floor(holdTime * gl.sampling_rate))
                success = dat.loc[tr].Success == 'success'
                X = force[start_sample:end_sample]
                force_pattern = force[end_sample:].mean(axis=0)
                MD, d = calc_md(X)
                ET = (end_sample - start_sample) / gl.sampling_rate
            elif Type=='TMS':
                forceAbs = np.abs(force[start_sample:])
                peakLoc = np.argmax(forceAbs[:50], axis=0) + start_sample
                force_pattern = force[peakLoc, np.arange(force.shape[1])]
                force_tms[tr, :, :] = force[:150].T
                MD = np.nan
                ET = np.nan
                success = np.nan
            for c, ch in enumerate(gl.channels):
                out_dict[ch].append(force_pattern[c])
            for c, ch in enumerate(gl.channels):
                out_dict[ch + '_PRE'].append(pre[c])
            out_dict['MD'].append(MD)
            out_dict['ET'].append(ET)
            out_dict['participant_id'].append(args.sn)
            out_dict['TN'].append(tr + 1)
            out_dict['BN'].append(dat.loc[tr].Block)
            out_dict['Chord'].append(Chord)
            out_dict['Chord_cluster'].append(Chord_cluster)
            out_dict['condition'].append(dat.loc[tr].Type)
            out_dict['session'].append('testing')
            out_dict['success'].append(success)
        df = pd.DataFrame(out_dict)
        df.to_csv(os.path.join(path, 'single_trial.tsv'), sep='\t', index=False)
        np.save(os.path.join(path, 'forceTMS.npy'), force_tms)



if __name__ == '__main__':

    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--sn', type=int, default=100)
    parser.add_argument('--sns', nargs='+', default=[101], type=int)

    args = parser.parse_args()

    main(args)

    end = time.time()
    print(f'Time elapsed: {end - start} seconds')