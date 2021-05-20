checkpoint_dir="./checkpoints_pirl"
log="pirl_logs.txt"

rm -rf $checkpoint_dir

model_config=pretrain/pirl/pirl_jigsaw_unet.yaml

python3 run_distributed_engines.py hydra.verbose=true \
    config=$model_config \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
    config.OPTIMIZER.num_epochs=100 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.CHECKPOINT.DIR=$checkpoint_dir >& $log

zip_out=checkpoints_pirl_"`date +"%m.%d.%Y-%H.%M.%S"`".zip
zip -r $zip_out $checkpoint_dir $log configs/config/$model_config
#aws s3 cp $zip_out s3://ylichman-dl-bucket/pirl/$zip_out

echo DONE DONE DONE
#sudo shutdown now
