import cv2 
import numpy as np
import imageio.v2 as iio
import os
import json
from PIL import Image
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from skimage.exposure import match_histograms


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def preprocess(img):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
        transforms.Normalize([123.675, 116.28, 103.53], [58.395, 57.12, 57.375]),
    ])
    return tf(img).unsqueeze(0)

def deprocess(img_tensor):
    x = img_tensor.clone()
    x = x.squeeze(0).cpu()
    x = x * torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
    x = x + torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    x = x.clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
    return x


def refine_edges_and_smooth(image, mask):
   
    blurred_mask = cv2.GaussianBlur(mask, (9, 9), sigmaX=2, sigmaY=2)

    edge_mask = cv2.Canny(blurred_mask, 50, 150) 
   
    edge_mask_3c = cv2.merge([edge_mask] * 3) / 255.0  

    blurred_image = cv2.GaussianBlur(image, (7, 7), sigmaX=1)

    out = image.astype(np.float32) * (1 - edge_mask_3c) + blurred_image.astype(np.float32) * edge_mask_3c

    return np.clip(out, 0, 255).astype(np.uint8)


def harmonize_insert_region_only(composite, reference, mask, strength=0.4):
    
    # mask: HxW, uint8 [0-255]
    composite = composite.astype(np.float32)
    reference = reference.astype(np.float32)
    mask = (mask / 255.0).astype(np.float32)[..., None]

    output = composite.copy()

    for c in range(3):
        src = composite[..., c]
        tgt = reference[..., c]

        # use the mask to compute means and stds
        src_mean = (src * mask[..., 0]).sum() / (mask[..., 0].sum() + 1e-6)
        src_std = np.sqrt(((src - src_mean)**2 * mask[..., 0]).sum() / (mask[..., 0].sum() + 1e-6))

        tgt_mean = (tgt * mask[..., 0]).sum() / (mask[..., 0].sum() + 1e-6)
        tgt_std = np.sqrt(((tgt - tgt_mean)**2 * mask[..., 0]).sum() / (mask[..., 0].sum() + 1e-6))

        adjusted = (src - src_mean) / src_std * tgt_std + tgt_mean
        adjusted = strength * adjusted + (1 - strength) * src

        # just apply to the masked region
        output[..., c] = output[..., c] * (1 - mask[..., 0]) + adjusted * mask[..., 0]

    return np.clip(output, 0, 255).astype(np.uint8)

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize_dim = (512, 512)
source_path = 'Test/data/test_video_3_source.png'
mask_path = 'Test/data/test_video_3_mask.png'
video_path = 'Test/Video/test_video_3.mp4'
homography_path = 'Test/data/test_video_3.json'
output_path = 'Test/Insertion_result/test_video_3.mp4'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load data
source_img = Image.open(source_path).convert('RGB').resize(resize_dim)
source_np = np.array(source_img)
mask_np = np.array(Image.open(mask_path).convert('L').resize(resize_dim))
reader = iio.get_reader(video_path)
fps = reader.get_meta_data()['fps']
frames = [np.array(Image.fromarray(f).resize(resize_dim)) for f in reader]
reader.close()
with open(homography_path, 'r') as f:
    homography_data = json.load(f)
homography_dict = {int(frame['frame_idx']): np.array(frame['homography_matrix'], dtype=np.float32) for frame in homography_data}

# new
ref_idx = 0

output_frames = []
for idx, frame_np in enumerate(tqdm(frames, desc="Stylizing Insertion")):
    H = homography_dict.get(idx)
    if H is None:
        output_frames.append(frame_np)
        continue

    warped_img = cv2.warpPerspective(source_np, H, resize_dim, flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(mask_np, H, resize_dim, flags=cv2.INTER_LINEAR)



    blended_np = frame_np * (1 - warped_mask[..., None] / 255.0) + warped_img * (warped_mask[..., None] / 255.0)
    
    blended_np = blended_np.astype(np.uint8)

    blended_np = harmonize_insert_region_only(blended_np, frame_np, mask=warped_mask, strength=0.4)

    blended_np = refine_edges_and_smooth(blended_np, warped_mask)
    
    output_frames.append(blended_np)

writer = iio.get_writer(output_path, fps=fps)
for frame in output_frames:
    writer.append_data(frame)
writer.close()
print("Saved stylized output to:", output_path)