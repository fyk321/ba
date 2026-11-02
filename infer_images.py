import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model import RefinementUNet
from datasets import SceneFlowDataset


def save_rgb_image(data, path):
    """Save an H×W×3 float32 array [0,1] as a lossless PNG at native resolution."""
    img = Image.fromarray((np.clip(data, 0, 1) * 255).astype(np.uint8))
    img.save(path)
    print(f"  Saved: {path}")


def save_disp_image(data, path, cmap='jet'):
    """Save a 2-D disparity array as a colormapped PNG at native resolution.

    plt.imsave writes exactly one output pixel per input element — no figure
    canvas, no DPI scaling, no interpolation.
    """
    plt.imsave(path, data, cmap=cmap)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Save SAD disparity, ground truth, original image, and model disparity as pure images"
    )

    # Dataset arguments
    parser.add_argument('--sceneflow_dir', type=str, required=True, help='SceneFlow root directory')
    parser.add_argument('--subdataset', type=str, default=None,
                        help='Filter SceneFlow samples by subdataset name, e.g. "driving" or "monkaa". '
                             '--sample_idx is then relative to the filtered subset.')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of the sample to visualize')

    # Output
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the 4 output images')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading SceneFlow sample from {args.sceneflow_dir}...")
    dataset = SceneFlowDataset(sceneflow_dir=args.sceneflow_dir)
    if args.subdataset:
        dataset.samples = [
            s for s in dataset.samples if args.subdataset in s[0]
        ]
        print(f"After filtering by '{args.subdataset}': {len(dataset.samples)} samples")
        assert len(dataset.samples) > 0, \
            f"No samples found matching subdataset '{args.subdataset}'"

    sample = dataset[args.sample_idx]

    # Model input shape is: (17, H, W).
    # First 14 channels are SAD disparities/costs, last 3 are Left RGB
    model_input = sample['input'].unsqueeze(0)    # (1, 17, H, W)
    target = sample['target'].squeeze(0).numpy()  # (H, W)
    valid_mask = sample['valid'].squeeze(0).numpy()  # (H, W)

    # Reconstruct RGB Left Image (last 3 channels of input)
    left_rgb = model_input[0, 18:].permute(1, 2, 0).numpy()

    # Extract one raw SAD disparity channel (channel index 2)
    # The first 7 channels are SAD disparities for different kernel sizes
    raw_sad = model_input[0, 2].numpy()

    # Mask invalid pixels in ground truth
    target_masked = np.ma.masked_where(valid_mask == 0, target)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = RefinementUNet(n_channels=21, n_classes=1).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        inputs = model_input.to(device)
        prediction = model(inputs)
        refined_disp = prediction.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)

    # Compute absolute difference between model prediction and ground truth
    # (only over valid pixels, consistent with diff.py)
    diff = np.abs(refined_disp - target)
    diff_masked = np.ma.masked_where(valid_mask == 0, diff)

    print("Saving pure images (no legends, captions or decorations)...")

    save_rgb_image(
        left_rgb,
        os.path.join(args.output_dir, "original_image.png")
    )

    save_disp_image(
        raw_sad,
        os.path.join(args.output_dir, "sad_disparity.png")
    )

    save_disp_image(
        refined_disp,
        os.path.join(args.output_dir, "model_disparity.png")
    )

    save_disp_image(
        target_masked,
        os.path.join(args.output_dir, "ground_truth_disparity.png")
    )

    save_disp_image(
        diff_masked,
        os.path.join(args.output_dir, "disparity_difference.png"),
        cmap='jet'
    )

    print("All images saved successfully.")


if __name__ == '__main__':
    main()
