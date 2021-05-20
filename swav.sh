checkpoint_dir="./checkpoints_swav"
log="swav_logs.txt"

rm -rf $checkpoint_dir

model_config=pretrain/swav/swav_8node_nestedunet_fastmri.yaml

python3 run_distributed_engines.py hydra.verbose=true \
    config=$model_config \
    config.OPTIMIZER.num_epochs=100 \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=16 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=4 \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
    config.CHECKPOINT.DIR=$checkpoint_dir >& $log

zip_out=checkpoints_swav_"`date +"%m.%d.%Y-%H.%M.%S"`".zip
zip -r $zip_out $checkpoint_dir $log configs/config/$model_config
#aws s3 cp $zip_out s3://ylichman-dl-bucket/swav/$zip_out

echo DONE DONE DONE
#sleep 15m
#sudo shutdown now
