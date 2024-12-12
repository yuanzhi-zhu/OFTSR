import torch
import os
import lpips
from PIL import Image
from torchvision import transforms
import tqdm
import argparse

def calc_psnr(batch1, batch2, max_pixel=2.0, eps=1e-10):
    mse = torch.mean((batch1 - batch2) ** 2, axis=(1, 2, 3))
    zeros = torch.zeros_like(mse)
    inf = torch.ones_like(mse) * float('inf')
    psnr_values = torch.where(mse == 0, inf, 20 * torch.log10(max_pixel / torch.sqrt(mse + eps)))
    psnr_values = torch.where(torch.isnan(psnr_values), zeros, psnr_values)
    mean_psnr = torch.mean(psnr_values)
    return mean_psnr.item()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fdir1 = '../DIV2K_valid_HR'
fdir1 = '../ffhq_val_100'
# fdir1 = './work_dir_DIVK_dis/sr_bicubic-sf4-guided_unet-t_uniform-sigma0.2-bs32-loss_l1-lr1e-05-amp_fp32-distil-v_boot-solver_rk2_0.5-dt0.05-w_distil_1.0-w_bound_0.1-w_align_0.01-no_gan-DIV2K_DIS/20241105-0223/eval_samples/20241105-2241HR'
fdir2 = './work_dir_DIVK_dis/sr_bicubic-sf4-guided_unet-t_uniform-sigma0.2-bs32-loss_l1-lr1e-05-amp_fp32-distil-v_boot-solver_rk2_0.5-dt0.05-w_distil_1.0-w_bound_0.1-w_align_0.01-no_gan-DIV2K_DIS/20241105-0223/eval_samples/20241105-1918'
# fdir2 = './results_ir-sde_DIV2k'
# fdir2 = './work_dir_DIVK_dis/sr_bicubic-sf4-guided_unet-t_uniform-sigma0.2-bs32-loss_l1-lr1e-05-amp_fp32-distil-v_boot-solver_rk2_0.5-dt0.05-w_distil_1.0-w_bound_0.1-w_align_0.01-no_gan-DIV2K_DIS/20241105-0223/eval_samples/20241105-2241'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir1", type=str, default=fdir1, help="Path to image folder 1.")
    # Additional arguments for overriding config
    parser.add_argument("--fdir2", type=str, default=fdir2, help="Path to image folder 2.")
    args = parser.parse_args()
    fdir1 = args.fdir1
    fdir2 = args.fdir2

    mode="alex"

    loss_fn_vgg = lpips.LPIPS(net=mode).to(device)

    image_list1 = [os.path.join(fdir1, img) for img in os.listdir(fdir1)]
    image_list2 = [os.path.join(fdir2, img) for img in os.listdir(fdir2)]
    image_list1.sort()
    image_list2.sort()

    lpips_score = 0
    psnr = 0
    ssim = 0

    image_transform = transforms.Compose([
        # transforms.CenterCrop(512),   # center crop
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    for img1, img2 in tqdm.tqdm(zip(image_list1, image_list2), total=len(image_list1)):
        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')
        img1 = image_transform(img1).unsqueeze(0).to(device)
        img2 = image_transform(img2).unsqueeze(0).to(device)
        lpips_score += loss_fn_vgg(img1, img2).item()
        psnr += calc_psnr(img1, img2)
    
    lpips_score /= len(image_list1)
    psnr /= len(image_list1)
    print(f'lpips: {lpips_score}, psnr: {psnr}')
    # python -m pytorch_fid ./work_dir_4/sr_bicubic-sf4-guided_unet-t_uniform-sigma0.2-bs32-loss_l1-lr2e-05-amp_fp32-ImageNet/20241022-2156/eval_samples/20241027-2149HR ./work_dir_4/sr_bicubic-sf4-guided_unet-t_uniform-sigma0.2-bs32-loss_l1-lr2e-05-amp_fp32-ImageNet/20241022-2156/eval_samples/20241027-2149 --batch-size 1