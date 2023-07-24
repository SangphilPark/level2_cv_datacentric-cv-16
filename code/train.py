import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import CustomSceneTextDataset
from model import EAST
from utils import split_folder, get_gt_bboxes, get_pred_bboxes
from deteval import *

import numpy as np
import random

import wandb
wandb.init(project="ocr")


def seed_everything(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '../data/medical'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])
    
    parser.add_argument('--seed', type=int, default=1110, help='random seed for train valid split')
    parser.add_argument('--finetune', type=bool, default=False, help='finetune or not. Default is training')

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, ignore_tags, seed, finetune):
    
    root_dir, train_images, valid_images = split_folder(seed=seed, test_size=0.2, data_dir=data_dir)

    train_dataset = CustomSceneTextDataset(
        root_dir,
        train_images,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags
    )
    
    train_dataset = EASTDataset(train_dataset)

    num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # finetuning 
    if finetune :
        model = EAST(pretrained=False).to(args.device)
        model.load_state_dict(torch.load('/opt/ml/input/code/trained_models/f1base_latest.pth', map_location='cpu'))
    
    # training
    else :   
        model = EAST() 
        
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    
    model.train()
    best_f1_score = 0
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 
                    'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        wandb.log({"train_mean_loss": epoch_loss / num_batches}, step=epoch)
        wandb.log({"train_Cls_loss": extra_info['cls_loss']}, step=epoch)
        wandb.log({"train_Angle_loss": extra_info['angle_loss']}, step=epoch)
        wandb.log({"train_IoU loss": extra_info['iou_loss']}, step=epoch)

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

        ckpt_fpath = osp.join(model_dir, 'latest.pth')
        torch.save(model.state_dict(), ckpt_fpath)
        
        # validation
        if (epoch+1) % 10 == 0 :
            with torch.no_grad():
                print("Calculating validation results...")
                
                pred_bboxes_dict = get_pred_bboxes(model, root_dir, valid_images, input_size, batch_size, split='valid',)            
                gt_bboxes_dict = get_gt_bboxes(root_dir, ufo_dir='/ufo/valid.json', valid_images=valid_images)
                
                result = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)
                total_result = result['total']
                precision, recall = total_result['precision'], total_result['recall']
                f1_score = 2*precision*recall/(precision+recall)
                print('F1 Score : {:.4f}'.format(f1_score))
                
                wandb.log({'valid_precision': precision}, step=epoch)
                wandb.log({'valid_recall': recall}, step=epoch)
                wandb.log({'valid_f1_score':f1_score}, step=epoch)
                
                if best_f1_score < f1_score:
                        print(f"New best model for f1 score : {f1_score}! saving the best model..")
                        bestpt_fpath = osp.join(model_dir, 'best.pth')
                        torch.save(model.state_dict(), bestpt_fpath)
                        best_f1_score = f1_score
                

def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    seed_everything(1)
    args = parse_args()
    main(args)