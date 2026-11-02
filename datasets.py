import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import re

def read_pfm(file_path):
    """
    Read a PFM file into a Numpy array. Note that it will have
    a shape of H x W, not W x H. Returns tuple containing the
    image and the scale factor from the file.
    """
    file = open(file_path, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

class SceneFlowDataset(Dataset):
    def __init__(self, sceneflow_dir):
        super().__init__()
        self.sceneflow_dir = sceneflow_dir
        self.samples = []  # list of (left_path, disp_path, sad_path)
        
        valid_img_exts = ('.png', '.jpg', '.jpeg', '.webp')
        
        # Find all cleanpass folders
        cleanpass_folders = sorted([
            f for f in os.listdir(sceneflow_dir)
            if os.path.isdir(os.path.join(sceneflow_dir, f)) and 'cleanpass' in f
        ])
        
        for cf in cleanpass_folders:
            cf_path = os.path.join(sceneflow_dir, cf)
            
            # Derive sibling folder names
            if 'cleanpass_webp' in cf:
                disp_folder = cf.replace('cleanpass_webp', 'disparity')
                sad_folder = cf.replace('cleanpass_webp', 'sad_disparity')
            else:
                disp_folder = cf.replace('cleanpass', 'disparity')
                sad_folder = cf.replace('cleanpass', 'sad_disparity')
            
            for root, dirs, files in os.walk(cf_path):
                if os.path.basename(root) != 'left':
                    continue
                    
                for file in sorted(files):
                    if not file.lower().endswith(valid_img_exts):
                        continue
                    
                    left_path = os.path.join(root, file)
                    
                    # Derive relative path from cleanpass folder
                    rel_path = os.path.relpath(left_path, cf_path)
                    
                    disp_rel = os.path.splitext(rel_path)[0] + '.pfm'
                    disp_path = os.path.join(sceneflow_dir, disp_folder, disp_rel)
                    
                    sad_rel = os.path.splitext(rel_path)[0] + '.npy'
                    sad_path = os.path.join(sceneflow_dir, sad_folder, sad_rel)
                    
                    # Only add if all three files exist
                    if os.path.exists(disp_path) and os.path.exists(sad_path):
                        self.samples.append((left_path, disp_path, sad_path))
        
        print(f"SceneFlowDataset: found {len(self.samples)} valid triplets in {sceneflow_dir}")
        assert len(self.samples) > 0, f"No valid triplets found in {sceneflow_dir}"

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        left_path, disp_path, sad_path = self.samples[idx]
        
        # Load Left Image (RGB)
        left_img = np.array(Image.open(left_path).convert('RGB'), dtype=np.float32) / 255.0
        
        # Load Ground Truth Disparity (PFM for SceneFlow)
        disp_img, _ = read_pfm(disp_path)
        disp_img = disp_img.astype(np.float32)
        valid_mask = (disp_img > 0).astype(np.float32)
        
        # Load SAD Disparities (precomputed numpy array shape: (18, H, W))
        sad_path = self.samples[idx][2]
        sad_disp = np.load(sad_path).astype(np.float32)
        
        # Convert to PyTorch tensors
        left_tensor = torch.from_numpy(left_img).permute(2, 0, 1)    # 3xHxW
        disp_tensor = torch.from_numpy(disp_img).unsqueeze(0)        # 1xHxW
        valid_tensor = torch.from_numpy(valid_mask).unsqueeze(0)     # 1xHxW
        sad_tensor = torch.from_numpy(sad_disp)                      # 18xHxW
        
        # Concatenate SAD (18 channels) + RGB (3 channels) = 21 channels
        model_input = torch.cat([sad_tensor, left_tensor], dim=0)
        
        return {
            'input': model_input,
            'target': disp_tensor,
            'valid': valid_tensor
        }
