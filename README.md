# OFTSR

This is the official implementation of the paper:
## [OFTSR: One-Step Flow for Image Super-Resolution with Tunable Fidelity-Realism Trade-offs](https://arxiv.org/abs/2412.09465)
by [Yuanzhi Zhu](https://yuanzhi-zhu.github.io/about/), [Ruiqing Wang](https://github.com/wrqcodedoge), [Shilin Lu](https://scholar.google.com/citations?user=gAG9WLYAAAAJ), [Hanshu Yan](https://hanshuyan.github.io/), [Junnan Li](https://scholar.google.com/citations?user=MuUhwi0AAAAJ), [Kai Zhang](https://cszn.github.io/)

## Training
########################## FFHQ Noiseless Restoration ##########################
```bash
python train_fm.py --opt configs/ir_fm_ffhq.yml \
        --overrides \
        dataset.val_path='./val_data/ffhq_val_100' \
        sample.num_sample=100 \
        ir.sigma_y=0. \
        ir.sigma_pertubation=0.1 \
        fm_model.use_cond=true
```

########################## FFHQ Noisy Restoration ##########################
```bash
python train_fm.py --opt configs/ir_fm_ffhq.yml \
        --overrides \
        dataset.val_path='./val_data/ffhq_val_100' \
        sample.num_sample=100 \
        ir.sigma_y=0.05 \
        ir.sigma_pertubation=0.5 \
        fm_model.use_cond=true
```

########################## DIV2K Noiseless Restoration ##########################
```bash
python train_fm.py --opt configs/ir_fm_DIV2K.yml \
        --overrides \
        dataset.val_path='./val_data/DIV2K_valid_HR' \
        sample.num_sample=100 \
        sample.psnr_batch_size=1 \
        ir.sigma_y=0. \
        ir.sigma_pertubation=0.2 \
        fm_model.use_cond=true
```

########################## ImageNet Noiseless Restoration ##########################
```bash
python train_fm.py --opt configs/ir_fm_imagenet.yml \
        --overrides \
        dataset.val_path='./val_data/imagenet_val_100' \
        sample.num_sample=100 \
        ir.sigma_y=0. \
        ir.sigma_pertubation=0.2 \
        fm_model.use_cond=true
```

## Sampling
########################## FFHQ Noiseless Restoration ##########################

#--------------------- multi-step ---------------------
```bash
python sample_fm.py --opt configs/ir_fm_ffhq.yml \
        --overrides \
        sample.pre_train_model=ckpts/sr_bicubic-sf4-guided_unet-sigma0.1-bs32-loss_l1-lr0.0001-FFHQ-checkpoint_10.pth \
        dataset.val_path='./val_data/ffhq_val_100' \
        sample.num_sample=100 \
        ir.sigma_y=0. \
        ir.sigma_pertubation=0.1 \
        fm_model.use_cond=true
```

#--------------------- one-step ---------------------
```bash
python sample_fm.py --opt configs/dis_fm_ffhq.yml \
        --overrides \
        sample.pre_train_model=ckpts/sr_bicubic-sf4-guided_unet-sigma0.1-bs32-loss_l1-lr1e-05-distil-v_boot-solver_rk2_0.5-dt0.05-w_distil_1.0-w_bound_0.1-w_align_0.01-FFHQ_DIS-checkpoint_2.pth \
        dataset.val_path='./val_data/ffhq_val_100' \
        sample.num_sample=100 \
        ir.sigma_y=0. \
        ir.sigma_pertubation=0.1 \
        fm_model.use_cond=true
        sample.one_step_t=0.99
```

########################## FFHQ Noisy Restoration ##########################

#--------------------- multi-step ---------------------
```bash
python sample_fm.py --opt configs/ir_fm_ffhq.yml \
        --overrides \
        sample.pre_train_model=ckpts/sr_avp-sf4-sigmay_0.05-guided_unet-sigma0.5-bs128-loss_l1-lr0.0001-FFHQ-checkpoint_23.pth \
        dataset.val_path='./val_data/ffhq_val_100' \
        sample.num_sample=100 \
        ir.sigma_y=0.05 \
        ir.sigma_pertubation=0.5 \
        fm_model.use_cond=true
```

#--------------------- one-step ---------------------
```bash
python sample_fm.py --opt configs/dis_fm_ffhq.yml \
        --overrides \
        sample.pre_train_model=ckpts/sr_avp-sf4-sigmay_0.05-guided_unet-sigma0.5-bs32-loss_l1-lr2e-05-distil-v_boot-solver_rk2_0.5-dt0.05-w_distil_1.0-w_bound_0.1-w_align_0.01-FFHQ_DIS-checkpoint_24.pth \
        dataset.val_path='./val_data/ffhq_val_100' \
        sample.num_sample=100 \
        ir.sigma_y=0.05 \
        ir.sigma_pertubation=0.5 \
        fm_model.use_cond=true
        sample.one_step_t=0.99
```

########################## DIV2K Noiseless Restoration ##########################

#--------------------- multi-step ---------------------
```bash
python sample_fm.py --opt configs/ir_fm_DIV2K.yml \
        --overrides \
        sample.pre_train_model=ckpts/sr_bicubic-sf4-guided_unet-sigma0.2-bs128-loss_l1-lr1e-05-DIV2K-checkpoint_4.pth \
        dataset.val_path='./val_data/DIV2K_valid_HR' \
        sample.num_sample=100 \
        sample.psnr_batch_size=1 \
        ir.sigma_y=0. \
        ir.sigma_pertubation=0.2 \
        fm_model.use_cond=true
```

#--------------------- one-step ---------------------
```bash
python sample_fm.py --opt configs/dis_fm_DIV2K.yml \
        --overrides \
        sample.pre_train_model=ckpts/sr_bicubic-sf4-guided_unet-sigma0.2-bs32-loss_l1-lr1e-05-distil-v_boot-solver_rk2_0.5-dt0.05-w_distil_1.0-w_bound_0.1-w_align_0.01-DIV2K_DIS-checkpoint_5.pth \
        dataset.val_path='./val_data/ffhq_val_100' \
        sample.num_sample=100 \
        sample.psnr_batch_size=1 \
        ir.sigma_y=0. \
        ir.sigma_pertubation=0.2 \
        fm_model.use_cond=true
        sample.one_step_t=0.99
```

########################## ImageNet Noiseless Restoration ##########################

#--------------------- multi-step ---------------------
```bash
python sample_fm.py --opt configs/ir_fm_imagenet.yml \
        --overrides \
        sample.pre_train_model=ckpts/sr_bicubic-sf4-guided_unet-sigma0.2-bs32-loss_l1-lr1e-04-ImageNet-checkpoint_10.pth \
        dataset.val_path='./val_data/imagenet_val_100' \
        sample.num_sample=100 \
        ir.sigma_y=0. \
        ir.sigma_pertubation=0.2 \
        fm_model.use_cond=true
```

#--------------------- one-step ---------------------
```bash
python sample_fm.py --opt configs/dis_fm_imagenet.yml \
        --overrides \
        sample.pre_train_model=ckpts/sr_bicubic-sf4-guided_unet-sigma0.2-bs8-loss_l1-lr1e-04-distil-v_boot-solver_rk2_0.5-dt0.05-w_distil_1.0-w_bound_0.1-w_align_0.01-ImageNet_DIS-checkpoint_10.pth \
        dataset.val_path='./val_data/imagenet_val_100' \
        sample.num_sample=100 \
        ir.sigma_y=0. \
        ir.sigma_pertubation=0.2 \
        fm_model.use_cond=true
        sample.one_step_t=0.99
```

## Checkpoints
checkpoints can be found here on HuggingFace: https://huggingface.co/Yuanzhi/OFTSR
To sample from these checkpoints, please follow the instructions in the README.md of the HuggingFace model.

## Citation
If you find this repo helpful, please cite:

```bibtex
@article{zhu2024oftsr,
  title={OFTSR: One-Step Flow for Image Super-Resolution with Tunable Fidelity-Realism Trade-offs},
  author={Zhu, Yuanzhi and Wang, Ruiqing and Lu, Shilin and Li, Junnan and Yan, Hanshu and Zhang, Kai},
  journal={arXiv preprint arXiv:2412.09465},
  year={2024}
}
```
