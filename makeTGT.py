import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

participant_id = 101
session = 'testing'
nBlock = 10

out = pd.DataFrame()

df = pd.read_csv('target/template.csv')

for BN in range(nBlock):

    # --- 1) Identify 5-row blocks (Rep==1 starts, must be Rep 1..5 and same Chord) ---
    starts = df.index[df["Rep"] == 1].tolist()
    blocks = []
    for s in starts:
        blk = df.loc[s:s+4].copy()
        if len(blk) != 5:
            raise ValueError(f"Block starting at {s} is not length 5.")
        if blk["Rep"].tolist() != [1, 2, 3, 4, 5]:
            raise ValueError(f"Block starting at {s} does not have Rep 1..5.")
        if blk["Chord"].nunique() != 1:
            raise ValueError(f"Block starting at {s} contains multiple Chords.")
        blocks.append(blk)

    # --- Shuffle block order (block integrity maintained) ---
    order = rng.permutation(len(blocks))
    out_tmp = pd.concat([blocks[i] for i in order], ignore_index=True)

    # --- 2) Fully shuffle Prep_dur across ALL rows ---
    prep = out_tmp["Prep_dur"].to_numpy().copy()
    rng.shuffle(prep)
    out_tmp["Prep_dur"] = prep

    # --- 3) Enforce Type constraint within each 5-row block ---
    # Rep1 always Chord; among Rep2-5 choose 2 or 3 to be TMS
    new_type = []
    for i in range(0, len(out_tmp), 5):
        types = ["Chord"] * 5
        n_tms = int(rng.choice([2, 3]))
        tms_pos = rng.choice([1, 2, 3, 4], size=n_tms, replace=False)  # positions for Rep2..5
        for p in tms_pos:
            types[p] = "TMS"
        new_type.extend(types)

    out_tmp["Type"] = new_type
    out_tmp["Participant"] = participant_id
    out_tmp["Block"] = BN+1

    if session == "training":
        out_tmp["Type"] = "Chord"

    out = pd.concat([out, out_tmp], ignore_index=True)

out["Trial"] = np.arange(1, len(out)+1)

out.to_csv(f"target/efcTMS_{session}_{participant_id}.csv", index=False)
