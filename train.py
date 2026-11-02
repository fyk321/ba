import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from datasets import SceneFlowDataset
from model import RefinementUNet

def smooth_l1_loss_valid(pred, target, valid_mask):
    """
    Computes Smooth L1 Loss only on valid pixels.
    """
    # pred and target are (B, 1, H, W)
    # valid_mask is (B, 1, H, W)
    
    loss = F.smooth_l1_loss(pred, target, reduction='none')
    
    # Apply valid mask (use torch.where to avoid NaN * 0 = NaN)
    loss = torch.where(valid_mask > 0, loss, torch.zeros_like(loss))
    
    # Average over valid pixels
    valid_count = valid_mask.sum()
    if valid_count > 0:
        return loss.sum() / valid_count
    else:
        return loss.sum() * 0.0

def calculate_metrics(pred, target, valid_mask):
    """
    Calculates End-Point-Error (EPE) and >3px error rate.
    """
    with torch.no_grad():
        abs_err = torch.abs(pred - target)
        abs_err = torch.where(valid_mask > 0, abs_err, torch.zeros_like(abs_err))
        
        valid_count = valid_mask.sum()
        
        if valid_count == 0:
            return 0.0, 0.0
            
        epe = abs_err.sum() / valid_count
        
        err_gt_3 = ((abs_err > 3) * valid_mask).sum() / valid_count
        
        return epe.item(), (err_gt_3 * 100.0).item()

def main():
    parser = argparse.ArgumentParser(description="Train Disparity Refinement CNN")
    
    # Dataset arguments
    parser.add_argument('--sceneflow_dir', type=str, required=True,
                        help='Root directory of SceneFlow dataset (contains *_cleanpass*, *_disparity, *_sad_disparity)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--log_dir', type=str, default='runs/experiment_1', help='TensorBoard log directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    
    # Pretrained checkpoint
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to a pretrained model checkpoint (.pth) to load before training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint (.pth) to resume training from')
    
    args = parser.parse_args()
    
    # --- Determine if running in DDP or single-GPU mode ---
    use_ddp = 'LOCAL_RANK' in os.environ
    
    if use_ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        assert torch.cuda.is_available(), "CUDA is required. No GPU detected."
        local_rank = 0
        global_rank = 0
        device = torch.device("cuda")
    
    is_main = (global_rank == 0)
    
    writer = None
    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
        if TENSORBOARD_AVAILABLE:
            writer = SummaryWriter(log_dir=args.log_dir)
    
    # --- Build dataset ---
    if is_main:
        print(f"Loading SceneFlow dataset from {args.sceneflow_dir}...")
    dataset = SceneFlowDataset(sceneflow_dir=args.sceneflow_dir)
    
    if use_ddp:
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                num_workers=1, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=1, pin_memory=True)
    
    # --- Build model ---
    model = RefinementUNet(n_channels=21, n_classes=1).to(device)
    
    # Load pretrained weights if specified
    if args.pretrained:
        if is_main:
            print(f"Loading pretrained weights from {args.pretrained}...")
        checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        if is_main:
            print("Pretrained weights loaded successfully.")
    
    if use_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])
        
        world_size = dist.get_world_size()
        base_global_batch = 64
        global_batch = world_size * args.batch_size
        lr_scale = global_batch / base_global_batch
        
        if lr_scale != 1.0:
            args.lr = args.lr * lr_scale
            if is_main:
                print(f"Auto-scaled learning rate by {lr_scale}x to {args.lr:.2e} for global batch size {global_batch}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Disable GradScaler when using bfloat16 to avoid scaling issues
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    
    best_epe = float('inf')
    global_step = 0
    start_epoch = 1
    
    if args.resume:
        if is_main:
            print(f"Resuming training from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_epe = checkpoint['best_epe']
            global_step = checkpoint.get('global_step', 0)
            
            if is_main:
                print(f"Successfully loaded resume checkpoint. Starting from epoch {start_epoch}")
        else:
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            import re
            m = re.search(r'model_epoch_(\d+)', args.resume)
            if m:
                start_epoch = int(m.group(1)) + 1
                if is_main:
                    print(f"Guessed epoch {start_epoch-1} from filename. Warning: Optimizer state was lost!")
            else:
                if is_main:
                    print("Loaded weights from old checkpoint. Could not guess epoch, starting from epoch 1.")

    if is_main:
        mode_str = "DDP" if use_ddp else "single-GPU"
        print(f"Starting training ({mode_str}) for {args.epochs} epochs on {device}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        if use_ddp:
            sampler.set_epoch(epoch)
        model.train()
        
        total_loss = torch.tensor(0.0, device=device)
        total_epe = torch.tensor(0.0, device=device)
        total_3px = torch.tensor(0.0, device=device)
        
        for step, batch in enumerate(dataloader, 1):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            valid_masks = batch['valid'].to(device)
            
            optimizer.zero_grad()
            
            # Use Automatic Mixed Precision (AMP) for a massive speedup on A100 Tensor Cores
            # bfloat16 avoids activation overflows that cause persistent NaNs in float16
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                preds = model(inputs)
                loss = smooth_l1_loss_valid(preds, targets, valid_masks)
            
            scaler.scale(loss).backward()
            
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epe, err_3px = calculate_metrics(preds, targets, valid_masks)
            
            total_loss += loss.detach()
            total_epe += epe
            total_3px += err_3px
            
            if is_main:
                if step % 20 == 0 or step == len(dataloader):
                    print(f"Epoch [{epoch}/{args.epochs}] - Step [{step}/{len(dataloader)}] - Loss: {loss.item():.4f}, EPE: {epe:.4f}")
                if writer is not None:
                    writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
                global_step += 1
                
        # Aggregate metrics
        num_batches = len(dataloader)
        
        if use_ddp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_epe, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_3px, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        else:
            world_size = 1
        
        avg_loss = (total_loss / num_batches) / world_size
        avg_epe = (total_epe / num_batches) / world_size
        avg_3px = (total_3px / num_batches) / world_size
        
        if is_main:
            if writer is not None:
                writer.add_scalar('Train/Loss_Epoch', avg_loss.item(), epoch)
                writer.add_scalar('Train/EPE', avg_epe.item(), epoch)
                writer.add_scalar('Train/3px_Error', avg_3px.item(), epoch)
            
            print(f"Epoch [{epoch}/{args.epochs}] - Loss: {avg_loss.item():.4f}, EPE: {avg_epe.item():.4f}, >3px Error: {avg_3px.item():.2f}%")
            
            # Save best model
            model_to_save = model.module if use_ddp else model
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_epe': best_epe,
                'global_step': global_step
            }
            
            if avg_epe.item() < best_epe:
                best_epe = avg_epe.item()
                save_dict['best_epe'] = best_epe
                save_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save(save_dict, save_path)
                print(f"Saved new best model to {save_path} with EPE: {best_epe:.4f}")
                
            # Save epoch model periodically
            if epoch % 10 == 0:
                torch.save(save_dict, os.path.join(args.save_dir, f'model_epoch_{epoch}.pth'))

    if is_main:
        if writer is not None:
            writer.close()
        print("Training complete.")
        
    if use_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
