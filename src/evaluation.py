# src/evaluate.py
# Standard library imports
import torch
import numpy as np
import pandas as pd

# Third-party imports
from tqdm import tqdm
from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.metric_stats import EER
import torch.nn.functional as F

def load_metadata(dataset_path, split):
    """Load metadata for a specific split."""
    metadata = pd.read_csv(
        f"{dataset_path}/metadata.csv", dtype={"baby_id": str, "chronological_index": str}
    )
    split_metadata = metadata.loc[metadata["split"] == split].copy()
    split_metadata["cry"] = split_metadata.apply(
        lambda row: read_audio(f'{dataset_path}/{row["file_name"]}').numpy(), axis=1
    )
    return split_metadata

def create_cry_dict(metadata):
    """Create a dictionary of concatenated cries for each (baby_id, period) group."""
    return pd.DataFrame(
        metadata.groupby(["baby_id", "period"])["cry"].agg(lambda x: np.concatenate(x.values)),
        columns=["cry"],
    ).to_dict(orient="index")

def get_embedding(encoder, model, data, args):
    feats = encoder.mods.compute_features(data)
    feats = encoder.mods.mean_var_norm(feats, torch.ones(data.shape[0]).to(args.device))
    embedding = model(feats)
    embedding = embedding.squeeze(1)
    return embedding

def encode_cries_with_segments(encoder, model, cry_dict, args, max_audio):
    """Encode the concatenated cries using the model with segments."""
    if max_audio is None:
        max_audio = 300 * 160 + 240 # 这里只是为了不报错
    model.eval()
    with torch.no_grad():
        for (baby_id, period), d in tqdm(cry_dict.items()):
            audio = d["cry"]
            data_1 = torch.FloatTensor(np.stack([audio], axis=0)).to(args.device)
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = np.pad(audio, (0, shortage), 'wrap')
            embedding_1 = get_embedding(encoder, model, data_1, args)
            embedding_2 = []
            scales = []
            scales.append(max_audio)
            for scale in scales:
                feats = []
                startframe = np.arange(0, audio.shape[0] - scale + 1, int(scale * (1 - args.overlap)))
                for asf in startframe:
                    feats.append(audio[int(asf):int(asf) + scale])
                feats = np.stack(feats, axis=0).astype(np.float32)
                data = torch.FloatTensor(feats).to(args.device)
                embedding = get_embedding(encoder, model, data, args)
                embedding_2.append(embedding)
            embedding_2 = torch.cat(embedding_2, dim=0)
                        
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            d["cry_encoded"] = [embedding_1, embedding_2]

def compute_cosine_similarity_score(row, cry_dict):
    """Compute the cosine similarity score between two encoded cries."""
    cos = torch.nn.CosineSimilarity(dim=-1)
    similarity_score = cos(
        cry_dict[(row["baby_id_B"], "B")]["cry_encoded"][0],
        cry_dict[(row["baby_id_D"], "D")]["cry_encoded"][0],
    )
    return similarity_score.item()

def compute_average_cosine_similarity_score(row, cry_dict, args):
    """Compute the average cosine similarity score for all possible pairs."""
    cos = torch.nn.CosineSimilarity(dim=-1)
    embedding_11, embedding_12 = cry_dict[(row['baby_id_B'], 'B')]['cry_encoded']
    embedding_21, embedding_22 = cry_dict[(row['baby_id_D'], 'D')]['cry_encoded']
    score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))
    
    matrix = torch.matmul(embedding_12, embedding_22.T)
    sorted_matrix = torch.sort(matrix.flatten())[0]
    score_2 = torch.mean(sorted_matrix)

    score = (score_1 + score_2) / 2
    return score.item()

def compute_eer_and_plot_verification_scores(pairs_df):
    """Compute EER and plot verification scores."""
    positive_scores = pairs_df.loc[pairs_df["label"] == 1]["score"].values
    negative_scores = pairs_df.loc[pairs_df["label"] == 0]["score"].values
    eer, threshold = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    return eer, threshold

from sklearn.metrics import roc_auc_score
def compute_auc(pair_df):
    """Compute the AUC score."""
    return roc_auc_score(pair_df["label"], pair_df["score"])

def evaluate(encoder, model, avg_state_dict, args):
    model.load_state_dict(avg_state_dict)
    model.to(args.device)

    # Load metadata and create cry dictionaries
    dev_metadata = load_metadata(args.dataset_path, "dev")
    test_metadata = load_metadata(args.dataset_path, "test")
    dev_cry_dict = create_cry_dict(dev_metadata)
    test_cry_dict = create_cry_dict(test_metadata)

    # Load pairs
    dev_pairs = pd.read_csv(f"{args.dataset_path}/dev_pairs.csv", dtype={"baby_id_B": str, "baby_id_D": str})
    test_pairs = pd.read_csv(f"{args.dataset_path}/test_pairs.csv")

    encode_cries_with_segments(encoder, model, test_cry_dict, args, max_audio=args.windows)
    test_pairs["score"] = test_pairs.apply(lambda row: compute_average_cosine_similarity_score(row, test_cry_dict, args), axis=1)
    test_eer_average, test_threshold_average_eer = compute_eer_and_plot_verification_scores(test_pairs)
    test_auc_average = compute_auc(test_pairs)

    test_pairs["score"] = test_pairs.apply(lambda row: compute_cosine_similarity_score(row, test_cry_dict), axis=1)
    test_eer_single, test_threshold_single_eer = compute_eer_and_plot_verification_scores(test_pairs)
    test_auc_single = compute_auc(test_pairs)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")
    print(f"Test-AUC (Single): {test_auc_single}")
    print(f"Test-AUC (Average): {test_auc_average}")
    print(f"Test-EER (Single): {test_eer_single}, Test-Threshold (Single): {test_threshold_single_eer}")
    print(f"Test-EER (Average): {test_eer_average}, Test-Threshold (Average): {test_threshold_average_eer}")