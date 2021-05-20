rm ./checkpoints_pirl/ -rf 
python3 tools/run_distributed_engines.py hydra.verbose=true \
    config=pretrain/pirl/pirl_jigsaw_unet.yaml \
    config.OPTIMIZER.num_epochs=1000 \
    config.CHECKPOINT.DIR="./checkpoints_pirl"