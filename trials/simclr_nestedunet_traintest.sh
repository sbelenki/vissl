rm -rf ./checkpoints_simclr

python3 tools/run_distributed_engines.py hydra.verbose=true \
    config=pretrain/simclr/simclr_4node_nestedunet_traintest.yaml \
    config.OPTIMIZER.num_epochs=100 \
    config.CHECKPOINT.DIR="./checkpoints_simclr"
