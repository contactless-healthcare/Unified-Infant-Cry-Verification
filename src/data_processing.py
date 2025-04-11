# src/data_processing.py

import pandas as pd
import numpy as np
from speechbrain.dataio.dataio import read_audio
import random
import pathlib
from speechbrain.dataio.dataio import write_audio
import torch


def create_cut_length_interval(row, cut_length_interval):
        """cut_length_interval is a tuple indicating the range of lengths we want our chunks to be.
        this function computes the valid range of chunk lengths for each audio file
        """
        # the lengths are in seconds, convert them to frames
        cut_length_interval = [round(length * 16000) for length in cut_length_interval]
        cry_length = round(row["duration"] * 16000)
        # make the interval valid for the specific sound file
        min_cut_length, max_cut_length = cut_length_interval
        # if min_cut_length is greater than length of cry, don't cut
        if min_cut_length >= cry_length:
            cut_length_interval = (cry_length, cry_length)
        # if max_cut_length is greater than length of cry, take a cut of length between min_cut_length and full length of cry
        elif max_cut_length >= cry_length:
            cut_length_interval = (min_cut_length, cry_length)
        return cut_length_interval

def data_processing(args):
    # read metadata csv and get the training split
    metadata = pd.read_csv(
        f"{args.dataset_path}/metadata.csv", dtype={"baby_id": str, "chronological_index": str}
    )
    train_metadata = metadata.loc[metadata["split"] == "train"].copy()

    # read the segments
    train_metadata["cry"] = train_metadata.apply(
        lambda row: read_audio(f'{args.dataset_path}/{row["file_name"]}').numpy(), axis=1
    )
    # concatenate all segments for each (baby_id, period) group
    manifest_df = pd.DataFrame(
        train_metadata.groupby(["baby_id"])["cry"].agg(lambda x: np.concatenate(x.values)),
        columns=["cry"],
    ).reset_index()
    # all files have 16000 sampling rate
    manifest_df["duration"] = manifest_df["cry"].apply(len) / 16000
    pathlib.Path(f"{args.dataset_path}/concatenated_audio_train").mkdir(exist_ok=True)
    manifest_df["file_path"] = manifest_df.apply(
        lambda row: f"{args.dataset_path}/concatenated_audio_train/{row['baby_id']}.wav",
        axis=1,
    )
    manifest_df.apply(
        lambda row: write_audio(
            filepath=f'{row["file_path"]}', audio=torch.tensor(row["cry"]), samplerate=16000
        ),
        axis=1,
    )
    manifest_df = manifest_df.drop(columns=["cry"])
    
    def split_by_period(row):
        if random.random() < 0.85:
            return "train"
        else:
            return "val"


    manifest_df["split"] = manifest_df.apply(
        lambda row: split_by_period(row), axis=1
    )

    # each instance will be identified with a unique id
    manifest_df["id"] = manifest_df["baby_id"]
    
    # 变长 or 定长
    if args.max_audio:
        manifest_df["cut_length_interval_in_frames"] = args.max_audio
    else:
        manifest_df["cut_length_interval_in_frames"] = manifest_df.apply(
            lambda row: create_cut_length_interval(row, cut_length_interval=args.cut_length_interval), axis=1
        )

    return manifest_df