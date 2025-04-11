# run.py
from src.dataset import AudioDataset
from src.model import create_model
from src.training import train_model
from torch.utils.data import DataLoader
from src.dataset import Sampler, collate_fn
from torch.utils.data import DataLoader

from src.data_processing import data_processing
from src.utils import set_seed
from src.evaluation import evaluate
import argparse
import torch
import time
import dill

def parse_tuple(value):
    return tuple(map(int, value.strip("()").split(",")))

parser = argparse.ArgumentParser(description = "trainer")
# ==================== 路径配置 ====================
parser.add_argument('--dataset_path', type=str, default="data",
                  help="数据集存放路径")
parser.add_argument('--save_path', type=str, default="experiments",
                  help="实验输出保存路径")
parser.add_argument('--pretrain_ecapa_path', type=str, default='pretrain/ecapa_tdnn.dill',
                  help="预训练ECAPA模型路径")

# ==================== 运行模式配置 ====================
parser.add_argument('--is_train', type=str, default="False",
                    choices=["True", "False", "true", "false"],
                    help="训练模式开关 (True/False)")
parser.add_argument('--model', type=str, default='PreTrainECAPA',
                  help="模型选择: XVEC/whisper/PreTrainECAPA")
parser.add_argument('--device', type=str, default='cuda:0',
                  help="训练设备 (cuda:0)")
parser.add_argument('--seed', type=int, default=2024,
                  help="随机种子(123/12345/2024/2025/42/...)")

# ==================== 数据配置 ====================
parser.add_argument('--batch_size', type=int, default=16,
                  help="批处理大小")
parser.add_argument('--feat_dim', type=int, default=80,
                  help="声学特征维度")

# ==================== 音频长度配置 ====================
# 训练长度模式 (二选一)
parser.add_argument('--max_audio', 
                  type=lambda x: None if str(x).lower() == 'none' else int(x),
                  default=600 * 160+240,  # 16000Hz采样率下6秒的采样点数+240缓冲
                  help="训练音频长度模式: "
                       "None-变长训练 | "
                       "int(如600 * 160+240)-定长训练(单位:采样点)")

parser.add_argument('--cut_length_interval', 
                  type=parse_tuple, 
                  default=(3, 5),
                  help="变长训练时的裁剪区间[秒]，当max_audio=None时生效 "
                       "(默认:3~5秒)")

# ==================== 训练超参数 ====================
parser.add_argument('--num_epochs', type=int, default=100,
                  help="训练轮次")
parser.add_argument('--lr', type=float, default=0.001,
                  help="学习率")
parser.add_argument('--margin_triplet', type=int, default=2,
                  help="Triplet loss边界值")
parser.add_argument('--embed_dim', type=int, default=192,
                  help="嵌入特征维度")
parser.add_argument('--best_model_num', type=int, default=5,
                  help="保留验证集性能最好的前N个模型参数(默认:5),"
                       "训练完成后会对这些模型的参数取平均，生成最终模型")
# ==================== 测试切片配置 ====================
parser.add_argument('--windows', 
                  type=int, 
                  default=600 * 160+240,
                  help="测试时的切片窗口大小[采样点] "
                       "(16000Hz下600 * 160+240=6秒)")

parser.add_argument('--overlap', 
                  type=float, 
                  default=0.75,
                  help="测试切片的重叠率 (范围:0-1, 默认:0.75即75%重叠)")

args = parser.parse_args()
set_seed(args.seed) 
args.is_train = args.is_train.lower() == "true"  # 转换为bool

def main():
    start_time = time.time()
    print('Model:', args.model)
    if args.max_audio is not None:
        print(f"Fix_Length: {(args.max_audio - 240) / 16000}s")
    else:
        print(f"Var_Length: {args.cut_length_interval}s")
    manifest_df = data_processing(args)
    dataset = AudioDataset(manifest_df, args=args)
    dataset_val = AudioDataset(manifest_df, args=args, split = 'val')
    dataloader_val = DataLoader(dataset_val,  batch_sampler=Sampler(batch_size = args.batch_size, data_len = dataset_val.__len__()), collate_fn=collate_fn)
    dataloader = DataLoader(dataset,  batch_sampler=Sampler(batch_size = args.batch_size, data_len = dataset.__len__()), collate_fn=collate_fn)

    model = create_model(args).to(args.device)
    with open(args.pretrain_ecapa_path, 'rb') as f:
        encoder = dill.load(f)

    if args.is_train:
        avg_state_dict = train_model(encoder, model, dataloader_val, dataloader, dataset, dataset_val, args)                
    else:
        if args.max_audio is not None:
           avg_state_dict = torch.load(f"{args.save_path}/{args.model}/epochs{args.num_epochs}_seed{args.seed}_{(args.max_audio-240)/16000}s.ckpt", map_location=args.device)
        else:
           avg_state_dict = torch.load(f"{args.save_path}/{args.model}/epochs{args.num_epochs}_seed{args.seed}.ckpt", map_location=args.device)
    
    evaluate(encoder, model, avg_state_dict, args)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time}")
    

if __name__ == '__main__':
    main()
