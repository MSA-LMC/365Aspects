import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import ast

class MultimodalDatasetForTrainT2(Dataset):
    def __init__(self, csv_file, audio_dir, video_dir, text_dir, question, label_col, rating_csv, args=None):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir  # Directory containing audio features
        self.video_dir = video_dir  # Directory containing video features
        self.text_dir = text_dir    # Directory containing text features
        self.question = question
        self.training_modal = args.modalities if args else None
        self.label_col = label_col
        self.rating = pd.read_csv(rating_csv)
        self.result_dict = {row['id']: row for _, row in self.rating.iterrows()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data.iloc[idx]['id']

        audio_files = []
        video_files = []
        text_files = []
        features = {}

        for q in self.question:
            audio_file = [f for f in os.listdir(self.audio_dir) if f.startswith(f"{sample_id}_{q}")]
            video_file = [f for f in os.listdir(self.video_dir) if f.startswith(f"{sample_id}_{q}")]
            text_file = [f for f in os.listdir(self.text_dir) if f.startswith(f"{sample_id}_{q}")]
            if not audio_file or not video_file or not text_file:
                raise FileNotFoundError(f"Missing modality for {sample_id}_{q}")
            audio_files.append(audio_file[0])
            video_files.append(video_file[0])
            text_files.append(text_file[0])

        # 不再平均池化，而是拼接为sequence
        if 'audio' in self.training_modal:
            features['audio'] = np.concatenate([np.expand_dims(np.load(os.path.join(self.audio_dir, f)), axis=0) for f in audio_files], axis=0)
        
        if 'video' in self.training_modal:
            features['video'] = np.concatenate([np.load(os.path.join(self.video_dir, f)) for f in video_files], axis=0)

        if 'text' in self.training_modal:
            features['text'] = np.concatenate([np.load(os.path.join(self.text_dir, f)) for f in text_files], axis=0)

        # audio_features = np.concatenate([np.load(os.path.join(self.audio_dir, f)) for f in audio_files], axis=0)
        # video_features = np.concatenate([np.expand_dims(np.load(os.path.join(self.video_dir, f)), axis=0) for f in video_files], axis=0)
        # text_features = np.concatenate([np.expand_dims(np.load(os.path.join(self.text_dir, f)), axis=0) for f in text_files], axis=0)

        # features = np.concatenate([audio_features, video_features, text_features], axis=-1)
        label_normalized = np.array([(self.result_dict[sample_id][col] - 1) / 4 for col in self.label_col])

        return {k: torch.tensor(v, dtype=torch.float32) for k, v in features.items()}, torch.tensor(label_normalized, dtype=torch.float32)

class MultimodalDatasetForTestT2(Dataset):
    def __init__(self, csv_file, audio_dir, video_dir, text_dir, question, rating_csv, args=None):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.text_dir = text_dir
        if isinstance(question, list) and len(question) == 1 and isinstance(question[0], str):
            # 将字符串形式的列表变成真正的 Python 列表
            self.question = ast.literal_eval(question[0])
        else:
            self.question = question
        self.question = question
        self.training_modal = args.modalities if args else None
        self.rating = pd.read_csv(rating_csv)

        self.result_dict = {}
        for _, row in self.rating.iterrows():
            key = row["id"]
            self.result_dict[key] = row

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data.iloc[idx]['id']
        audio_files, video_files, text_files = [], [], []
        features = {}

        for q in self.question:
            audio_file = [f for f in os.listdir(self.audio_dir) if f.startswith(f"{sample_id}_{q}")]
            video_file = [f for f in os.listdir(self.video_dir) if f.startswith(f"{sample_id}_{q}")]
            text_file = [f for f in os.listdir(self.text_dir) if f.startswith(f"{sample_id}_{q}")]

            if not audio_file or not video_file or not text_file:
                raise FileNotFoundError(f"Files for {sample_id}_{q} not found.")
            audio_files.append(audio_file[0])
            video_files.append(video_file[0])
            text_files.append(text_file[0])
        
        if 'audio' in self.training_modal:
            features['audio'] = np.concatenate([np.expand_dims(np.load(os.path.join(self.audio_dir, f)), axis=0) for f in audio_files], axis=0)
        
        if 'video' in self.training_modal:
            features['video'] = np.concatenate([np.load(os.path.join(self.video_dir, f)) for f in video_files], axis=0)

        if 'text' in self.training_modal:
            features['text'] = np.concatenate([np.load(os.path.join(self.text_dir, f)) for f in text_files], axis=0)

        return {k: torch.tensor(v, dtype=torch.float32) for k, v in features.items()}, sample_id
    



def collate_fn_train(batch):
    features_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])

    features = {}
    masks = {}
    for k in features_list[0].keys():
        modality_tensors = [f[k] for f in features_list]

        if k == 'audio':
            modality_tensors = [item[0][k] for item in batch]
            lengths = [t.shape[0] for t in modality_tensors]
            max_len = max(lengths)
            padded = pad_sequence(modality_tensors, batch_first=True)
            mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
            audio_mask = mask.float()
            features[k] = padded
            masks[k + "_mask"] = audio_mask
        else:
            features[k] = torch.stack(modality_tensors)

    return features, masks, labels


def collate_fn_test(batch):
    # batch: List of (features_dict, sample_id)
    features_list = [item[0] for item in batch]
    sample_ids = [item[1] for item in batch]

    features = {}
    masks = {}

    for k in features_list[0].keys():
        modality_tensors = [f[k] for f in features_list]

        if k == 'audio':
            lengths = [t.shape[0] for t in modality_tensors]
            max_len = max(lengths)
            padded = pad_sequence(modality_tensors, batch_first=True)  # (B, T, D)
            mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
            audio_mask = mask.float()
            features[k] = padded
            masks[k + "_mask"] = audio_mask
        else:
            features[k] = torch.stack(modality_tensors)

    return features, masks, sample_ids