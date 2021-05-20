checkpoint_dir="./checkpoints_simclr"
log="simclr_logs.txt"
epochs=200
rm -rf $checkpoint_dir

model_config=pretrain/simclr/simclr_4node_unet_traintest.yaml
python3 run_distributed_engines.py hydra.verbose=true \
    config=$model_config \
    config.OPTIMIZER.num_epochs=$epochs \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
    config.CHECKPOINT.DIR=$checkpoint_dir >& $log

zip_out=checkpoints_simclr_${epochs}ep_"`date +"%m.%d.%Y-%H.%M.%S"`".zip
zip -r $zip_out $checkpoint_dir $log configs/config/$model_config
#aws s3 cp $zip_out s3://ylichman-dl-bucket/simclr/$zip_out

echo DONE DONE DONE
#sleep 15m
#sudo shutdown now
