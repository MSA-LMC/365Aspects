# 重构后的 train_task2.py

import os
import argparse
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
# from dataset.baseline_dataset2 import MultimodalDatasetForTrainT2
# from dataset.baseline_dataset2 import collate_fn_train, collate_fn_test
from dataset.baseline_dataset2_vote import MultimodalDatasetForTrainT2, MultimodalDatasetForTestT2
from dataset.baseline_dataset2_vote import collate_fn_train, collate_fn_test
from tqdm import tqdm, trange
# from model.new_model.builder import FusionModel
from model.vote_model.M_model import SharedMLPwEnsemble
import json
from datetime import datetime

# def train_model(model, train_loader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0
#     for features, labels in train_loader:
#         features, labels = features.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(features)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(train_loader)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    train_bar = tqdm(train_loader, desc="Training", leave=False)
    for features, mask, labels in train_bar:
        features = {k: v.to(device) for k, v in features.items()}
        audio_feat = features['audio']
        video_feat = features['video']
        text_feat = features['text']
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(audio_feat, video_feat, text_feat)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())
    return total_loss / len(train_loader)


# def evaluate_model(model, loader, criterion, device):
#     model.eval()
#     total_loss, predictions, targets = 0, [], []
#     with torch.no_grad():
#         for features, labels in loader:
#             features, labels = features.to(device), labels.to(device)
#             outputs = model(features)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             predictions.append(outputs.cpu().numpy())
#             targets.append(labels.cpu().numpy())
#     return total_loss / len(loader), mean_squared_error(np.concatenate(targets), np.concatenate(predictions))


def test_model(model, test_loader, device, columns, output_csv_path, test_csv_path):
    model.eval()
    predictions = []
    test_bar = tqdm(test_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for features, mask, ids in test_bar:
            # Unpack the tuple: features and mask
            features = {k: v.to(device) for k, v in features.items()}
            audio_feat = features['audio']
            video_feat = features['video']
            text_feat = features['text']
            outputs = model(audio_feat, video_feat, text_feat)
            outputs = outputs * 4 + 1
            predictions.append(outputs.squeeze().cpu().numpy())
        
        all_predictions = np.concatenate(predictions)
        # 读取验证集 CSV 文件
        val_df = pd.read_csv(test_csv_path)
        # 只保留第一列id
        val_df = val_df.iloc[:, [0]]
        pred_df = pd.DataFrame(all_predictions, columns=columns)
        # 合并原始数据和预测数据
        result_df = pd.concat([val_df, pred_df], axis=1)
        result_df.to_csv(output_csv_path, index=False)
        print(f"Test predictions saved to {output_csv_path}")


def evaluate_model(model, loader, criterion, device, columns, is_Test=False):
    model.eval()
    total_loss, predictions, targets = 0, [], []
    dim_losses = [0] * len(columns)
    val_bar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for features, mask, labels in val_bar:
            features = {k: v.to(device) for k, v in features.items()}
            audio_feat = features['audio']
            video_feat = features['video']
            text_feat = features['text']
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            outputs = model(audio_feat, video_feat, text_feat)
            
            # Combined loss for all dimensions
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions.append(outputs.squeeze().cpu().numpy())
            targets.append(labels.cpu().numpy())
            
            # Calculate per-dimension losses when not in training
            if is_Test:
                for i in range(len(columns)):
                    dim_loss = criterion(outputs[:, i], labels[:, i])
                    dim_losses[i] += dim_loss.item()
            
            val_bar.set_postfix(val_loss=loss.item())
        
        all_predictions = np.concatenate(predictions)
        all_targets = np.concatenate(targets)
        overall_mse = mean_squared_error(all_targets, all_predictions)
        if not is_Test:
            return total_loss / len(loader), overall_mse
        else:
            # Calculate per-dimension MSE and normalize dimension losses
            dim_mse = [mean_squared_error(all_targets[:, i], all_predictions[:, i]) for i in range(len(columns))]
            dim_losses = [loss / len(loader) for loss in dim_losses]
            return total_loss / len(loader), overall_mse, dim_losses, dim_mse

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def save_args(args, save_dir="./args_log"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args_file = os.path.join(save_dir, f"args_{timestamp}.json")
    args_file1 = os.path.join(args.log_dir, f"args_{timestamp}.json")
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4)
    with open(args_file1, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"📝 Args saved to {args_file},{args_file1}")

def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Make the behavior deterministic
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior in convolutions


def main():
    init_seed(seed=42)
    parser = argparse.ArgumentParser()
    #### for dataset
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--label_col', nargs='+', required=True)  # Integrity Collegiality Social_versatility Development_orientation Hireability
    parser.add_argument('--question', nargs='+', required=True)   # q1 q2 q3 q4 q5
    parser.add_argument('--rating_csv', required=True)

    #### for input_features
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--text_dir', required=True)
    parser.add_argument('--audio_dim', type=int, default=384)
    parser.add_argument('--video_dim', type=int, default=512)
    parser.add_argument('--text_dim', type=int, default=768)
    
    #### for training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool,default=True)
    parser.add_argument('--optim', type=str, default='adamw')

    #### for testing
    parser.add_argument('--only_test', action='store_true', default=False, help='Only run testing phase')
    parser.add_argument('--test_output_csv', type=str, default='test_predictions.csv')
    parser.add_argument('--test_model', default='best_model.pth')


    #### for model
    # for projector
    parser.add_argument('--HCPdropout_audio', type=float, default=0.2)
    parser.add_argument('--HCPdropout_video', type=float, default=0.2)
    parser.add_argument('--HCPdropout_text', type=float, default=0.2)
    parser.add_argument('--HCPdropout_pure_text', type=float, default=0.1) 
    parser.add_argument('--use_prompt', type=bool, default=False) # 可学习的prompt
    parser.add_argument('--unified_dim', type=int, default=768)   # projector对齐各个模态后的维度
    # for AT_VT connector
    parser.add_argument('--heads_num', type=int, default=4)     
    parser.add_argument('--ATCdropout', type=float, default=0.3)  # AT跨模态交互模块的dropout
    parser.add_argument('--VTCdropout', type=float, default=0.3)  # VT跨模态交互模块的dropout
    parser.add_argument('--hidden_dim', type=int, default=256)    # 三模态进入text增强器前的维度，也是text增强器的输入维度
    # for text feature enhancer
    parser.add_argument('--enhancer_dim', type=int, default=512)  # text增强器的输出维度
    parser.add_argument('--TFEdropout', type=float, default=0.2)  # text增强器的dropout
    # for regression head
    parser.add_argument('--RHdropout', type=float, default=0.2)   # 回归头的dropout
    parser.add_argument('--target_dim', type=int, default=5)      # 最终回归的维度
    parser.add_argument('--num_modalities', type=int, default=3)
    parser.add_argument('--modalities', type=str, default="audio,video,text")
    parser.add_argument('--output_model', default='best_model.pth')
    parser.add_argument('--loss_plot_path', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--training_time', type=str)
    args = parser.parse_args()
    args.modalities = [m.strip() for m in args.modalities.split(',')]
    save_args(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = MultimodalDatasetForTrainT2(args.train_csv, args.audio_dir, args.video_dir, args.text_dir, args.question, args.label_col, args.rating_csv,args)
    val_set = MultimodalDatasetForTrainT2(args.val_csv, args.audio_dir, args.video_dir, args.text_dir, args.question, args.label_col, args.rating_csv, args)
    test_set = MultimodalDatasetForTestT2(args.test_csv, args.audio_dir, args.video_dir, args.text_dir, args.question, args.rating_csv, args)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,collate_fn=collate_fn_train, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_train)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_test)

    # model = FusionModel(args).to(device)
    model = SharedMLPwEnsemble(args).to(device)
    criterion = nn.MSELoss()

    if args.only_test:
        model.load_state_dict(torch.load(args.test_model))
        model.eval()
        test_model(model, test_loader, device, args.label_col, output_csv_path=args.test_output_csv, test_csv_path='data/test_data_basic_information.csv')
        test_loss, test_mse, dim_losses, dim_mse = evaluate_model(model, val_loader, criterion, device, args.label_col, is_Test=True)
        print(f"Overall Test Loss: {test_loss*16:.4f}, Overall Test MSE: {test_mse*16:.4f}")
        print("\nPer-dimension metrics:")
        dimension_names = args.label_col
        for i, (name, loss, mse) in enumerate(zip(dimension_names, dim_losses, dim_mse)):
            print(f"{name:<25} Loss: {loss*16:.4f}, MSE: {mse*16:.4f}")
    
        return

    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Training started...")
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in trange(args.num_epochs, desc="Epochs", ncols=100):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_mse = evaluate_model(model, val_loader, criterion, device, args.label_col, is_Test=False)
        p_train_loss = train_loss * 16
        p_val_loss = val_loss * 16
        p_val_mse = val_mse * 16
    
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, args.output_model)

        tqdm.write(f"[Epoch {epoch+1}/{args.num_epochs}] "
               f"Train Loss: {p_train_loss:.4f} | Val Loss: {p_val_loss:.4f} | Val MSE: {p_val_mse:.4f}")

    # best_val_loss = float('inf')
    # for epoch in range(args.num_epochs):
    #     train_loss = train_model(model, train_loader, criterion, optimizer, device)
    #     val_loss, val_mse = evaluate_model(model, val_loader, criterion, device)
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         save_model(model, args.output_model)
    #     print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MSE={val_mse:.4f}")

    # 保存loss曲线
    save_loss_plot(train_losses, val_losses, args.loss_plot_path)
   
    model.load_state_dict(torch.load(args.output_model))
    model.eval()
    test_loss, test_mse, dim_losses, dim_mse = evaluate_model(model, val_loader, criterion, device, args.label_col, is_Test=True)
    test_model(model, test_loader, device, args.label_col, output_csv_path=args.test_output_csv, test_csv_path='data/test_data_basic_information.csv')
    print(f"Overall Test Loss: {test_loss*16:.4f}, Overall Test MSE: {test_mse*16:.4f}")
    print("\nPer-dimension metrics:")
    dimension_names = args.label_col
    for i, (name, loss, mse) in enumerate(zip(dimension_names, dim_losses, dim_mse)):
        print(f"{name:<25} Loss: {loss*16:.4f}, MSE: {mse*16:.4f}")
    
    

if __name__ == '__main__':
    main()
