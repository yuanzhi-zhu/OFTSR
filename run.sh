DID=0
IFS=',' read -r -a array <<< "$DID"
CUDA_VISIBLE_DEVICES=$DID torchrun --standalone --nproc_per_node=${#array[@]} \
    train_fm.py --opt configs/ir_fm_DIV2K.yml \

# CUDA_VISIBLE_DEVICES=2 python sample_fm_unified.py --opt configs/ir_fm_imagenet_unified.yml \
#         --overrides \
#         dataset.val_path='testdata/Val_SR/gt' \
#         sample.pre_train_model=ckpts/sr_bicubic-sf4-guided_unet-sigma0.2-bs32-loss_l1-lr1e-04-ImageNet-checkpoint_10.pth \
#         ir.sigma_pertubation=0.2 \
#         ir.degradation='sr_interp' \
#         fm_model.use_cond=true \
#         sample.one_step_t=0.99