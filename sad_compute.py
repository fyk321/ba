import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import argparse
import multiprocessing as mp

def compute_sad_disparity(left_img, right_img, max_disp=192, kernel_sizes=[3, 5, 7, 11, 15, (7, 3), (3, 7)], device='cuda'):
    """
    Computes SAD disparity for given kernel sizes.
    left_img, right_img: torch.Tensor of shape (1, C, H, W)
    Returns: numpy array of shape (len(kernel_sizes), H, W)
    """
    left_img = left_img.to(device)
    right_img = right_img.to(device)
    
    _, C, H, W = left_img.shape
    
    padded_right = F.pad(right_img, (max_disp, 0, 0, 0), mode='constant', value=0)
    
    # Cost volume size: (max_disp, 1, H, W)
    cost_volume = torch.zeros((max_disp, 1, H, W), device=device)
    
    for d in range(max_disp):
        shifted_right = padded_right[:, :, :, max_disp - d : max_disp - d + W]
        cost_volume[d, 0] = torch.abs(left_img[0] - shifted_right[0]).sum(dim=0)
    cost_volume_reshaped = cost_volume.transpose(0, 1) # (1, max_disp, H, W)
    
    disparities = []
    sad_values = []
    
    second_best_disparities = []
    second_best_sad_values = []
    
    for k in kernel_sizes:
        if isinstance(k, tuple):
            kh, kw = k
        else:
            kh, kw = k, k
            
        padding = (kh // 2, kw // 2)
        weights = torch.ones((max_disp, 1, kh, kw), device=device)
        
        # Sum over kh x kw window
        sad_cost = F.conv2d(cost_volume_reshaped, weights, padding=padding, groups=max_disp)[0] # (max_disp, H, W)
        
        if k in [3, 5]:
            top2_sads, top2_disps = torch.topk(sad_cost, k=2, dim=0, largest=False)
            best_sad, best_disp = top2_sads[0], top2_disps[0]
            
            second_sad, second_disp = top2_sads[1], top2_disps[1]
            second_best_disparities.append(second_disp.to(torch.float32).cpu().numpy())
            
            normalized_second_sad = second_sad.to(torch.float32) / (kh * kw * 3.0)
            second_best_sad_values.append(normalized_second_sad.cpu().numpy())
        else:
            # Find minimum SAD cost and its index (which is the disparity)
            best_sad, best_disp = torch.min(sad_cost, dim=0)
        
        best_disp = best_disp.to(torch.float32)
        disparities.append(best_disp.cpu().numpy())
        
        # Normalize SAD value by the number of pixels in the kernel and 3 color channels
        normalized_sad = best_sad.to(torch.float32) / (kh * kw * 3.0)
        sad_values.append(normalized_sad.cpu().numpy())
        
    disparities.extend(second_best_disparities)
    sad_values.extend(second_best_sad_values)
        
    # Concatenate disparities and normalized SAD values
    return np.concatenate([np.stack(disparities, axis=0), np.stack(sad_values, axis=0)], axis=0)
def process_pair(args):
    left_path, right_path, out_path, max_disp, kernels, device = args
    expected_channels = (len(kernels) + sum(1 for k in kernels if k in [3, 5])) * 2
    if os.path.exists(out_path):
        try:
            existing_shape = np.load(out_path, mmap_mode='r').shape
            if existing_shape[0] == expected_channels:
                return True
        except:
            pass
        
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        left_img = np.array(Image.open(left_path).convert('RGB'), dtype=np.float32) / 255.0
        right_img = np.array(Image.open(right_path).convert('RGB'), dtype=np.float32) / 255.0
        
        # Convert to NCHW
        left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).unsqueeze(0)
        right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            sad_disp = compute_sad_disparity(left_tensor, right_tensor, max_disp=max_disp, kernel_sizes=kernels, device=device)
        
        np.save(out_path, sad_disp)
        del left_tensor, right_tensor, sad_disp
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"Error processing {left_path}: {e}")
        return False

def parse_kernels(s):
    if 'x' in s:
        h, w = s.split('x')
        return (int(h), int(w))
    return int(s)

def main():
    parser = argparse.ArgumentParser(description="Precompute SAD disparities")
    parser.add_argument('--left_dir', type=str, help='Directory containing left images')
    parser.add_argument('--right_dir', type=str, help='Directory containing right images')
    parser.add_argument('--output_dir', type=str, help='Directory to save output numpy files')
    parser.add_argument('--sceneflow_dir', type=str, help='Root directory of SceneFlow (contains driving_cleanpass_webp, etc.)')
    parser.add_argument('--output_dir_suffix', type=str, default='sad_disparity', help='Suffix for the output directory when using --sceneflow_dir')
    parser.add_argument('--max_disp', type=int, default=192, help='Maximum disparity')
    parser.add_argument('--kernels', type=parse_kernels, nargs='+', default=[3, 5, 7, 11, 15, (7, 3), (3, 7)], help='Kernel sizes (e.g. 3 or 7x3)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    assert torch.cuda.is_available(), "CUDA is required. No GPU detected."
    
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    pairs = []
    
    if args.sceneflow_dir:
        cleanpass_folders = [f for f in os.listdir(args.sceneflow_dir) 
                             if os.path.isdir(os.path.join(args.sceneflow_dir, f)) and 'cleanpass' in f]
        
        for cf in cleanpass_folders:
            cf_path = os.path.join(args.sceneflow_dir, cf)
            for root, dirs, files in os.walk(cf_path):
                if os.path.basename(root) == 'left':
                    right_dir = os.path.join(os.path.dirname(root), 'right')
                    if os.path.isdir(right_dir):
                        for file in files:
                            if file.lower().endswith(valid_exts):
                                left_path = os.path.join(root, file)
                                right_path = os.path.join(right_dir, file)
                                if os.path.exists(right_path):
                                    
                                    # Output path logic
                                    rel_path = os.path.relpath(left_path, args.sceneflow_dir)
                                    parts = rel_path.split(os.sep)
                                    
                                    if 'cleanpass_webp' in parts[0]:
                                        parts[0] = parts[0].replace('cleanpass_webp', args.output_dir_suffix)
                                    elif 'cleanpass' in parts[0]:
                                        parts[0] = parts[0].replace('cleanpass', args.output_dir_suffix)
                                        
                                    out_rel = os.path.join(*parts)
                                    out_rel = os.path.splitext(out_rel)[0] + '.npy'
                                    out_path = os.path.join(args.sceneflow_dir, out_rel)
                                    
                                    pairs.append((left_path, right_path, out_path))
    else:
        assert args.left_dir and args.right_dir and args.output_dir,         os.makedirs(args.output_dir, exist_ok=True)
        left_files = sorted([f for f in os.listdir(args.left_dir) if f.lower().endswith(valid_exts)])
        right_files = sorted([f for f in os.listdir(args.right_dir) if f.lower().endswith(valid_exts)])
        assert len(left_files) == len(right_files), "Mismatch in number of left and right images"
        
        for lf, rf in zip(left_files, right_files):
            left_path = os.path.join(args.left_dir, lf)
            right_path = os.path.join(args.right_dir, rf)
            out_name = os.path.splitext(lf)[0] + '.npy'
            out_path = os.path.join(args.output_dir, out_name)
            pairs.append((left_path, right_path, out_path))
            
    print(f"Found {len(pairs)} stereo pairs to process.", flush=True)
    
    num_gpus = torch.cuda.device_count()
        
    mp_args = []
    for i, (left_path, right_path, out_path) in enumerate(pairs):
        device = f'cuda:{i % num_gpus}'
        mp_args.append((left_path, right_path, out_path, args.max_disp, args.kernels, device))
        
    mp.set_start_method('spawn', force=True)
    num_workers = min(mp.cpu_count(), len(pairs), num_gpus * 4)
    
    print(f"Processing using {num_workers} workers across {num_gpus} GPUs...", flush=True)
    
    with mp.Pool(num_workers) as pool:
        # Process all pairs in parallel and exhaust the generator
        for idx, _ in enumerate(pool.imap_unordered(process_pair, mp_args), 1):
            if idx % 100 == 0 or idx == len(mp_args):
                print(f"Processed {idx}/{len(mp_args)} pairs...", flush=True)
        
if __name__ == "__main__":
    main()
