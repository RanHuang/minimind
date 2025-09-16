import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from transformers import TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset
import datetime
import random
import socket

warnings.filterwarnings('ignore')

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def Logger(content):
    if not ddp or dist.get_rank() == 0:
        logger.info(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, swanlab):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (swanlab is not None) and (not ddp or dist.get_rank() == 0):
                swanlab.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('/data00/train/doudou/tokenizer')
    model = MiniMindForCausalLM(lm_config).to(args.device)
    logger.info(model)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

def generate_and_print_sample(model, tokenizer):
    model.eval()
    
    # 添加调试信息
    device = next(model.parameters()).device
    logger.info(f"Model device: {device}")
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    messages = ['人类大脑的主要功能', '世界上最高的山峰是', '万有引力原理是']
    message = messages[random.randint(0, len(messages) - 1)]
    logger.info(f"Generating sample for prompt: {message}")
    prompt = tokenizer.bos_token + message

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device=device)
    
    # 添加输入调试信息
    logger.info(f"Input shape: {inputs['input_ids'].shape}")
    logger.info(f"Input device: {inputs['input_ids'].device}")
    logger.info(f"Input dtype: {inputs['input_ids'].dtype}")
    
    generated_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=256,
        num_return_sequences=1,
        do_sample=True,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2,
        min_new_tokens=20,
        early_stopping=False
    )
    
    # 添加生成结果调试信息
    logger.info(f"Generated shape: {generated_ids.shape}")
    # logger.info(f"Generated text: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}")

    model.train()

def environment_info():
    # 获取节点IP
    node_ip = socket.gethostbyname(socket.gethostname())
    logger.info(f"Node IP: {node_ip}")
    # 获取GPU信息
    if torch.cuda.is_available():
        logger.info(f"GPU Count: {torch.cuda.device_count()}; GPU Information:")
        logger.info(torch.cuda.get_device_properties(0))
    else:
        logger.info("No GPU available.")

# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="/data00/train/doudou/pretrain")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument("--swanlab_project", type=str, default="minimind")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=16, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="/data00/dataset/minimind_dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    gpu_count = torch.cuda.device_count()
    args.swanlab_run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d')}-MiniMind-Pretrain-{gpu_count}-GPU"
    if args.use_swanlab and (not ddp or ddp_local_rank == 0):
        logger.info(f"SwanLab run name: {args.swanlab_run_name}")
        import swanlab
        swanlab.init(project=args.swanlab_project, name=args.swanlab_run_name)
    else:
        swanlab = None

    environment_info()

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers,
        num_key_value_heads=None,
        use_moe=args.use_moe)
    logger.info(f"LLM Config: {lm_config}")
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    model, tokenizer = init_model(lm_config)
    generate_and_print_sample(model, tokenizer)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Batch size: {args.batch_size}")

    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    begin_time = time.time()

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, swanlab)
    
    generate_and_print_sample(model, tokenizer)

    end_time = time.time()
    logger.info(f"Total time: {(end_time - begin_time) // 60:.3f}min")
