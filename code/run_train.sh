#/bin/bash

NAME=bert-base_k8:2_lr5e-5_warm0_wd_0.01_seed452_4
python train.py --run_name 10/1_$NAME --train_batch_size 32 \
        --num_train_epochs 10 --learning_rate 5e-5 --warmup_steps 0 --weight_decay 0.01 \
        --output_dir ./results/$NAME --random_seed 452

#NAME=bert-base_k8:2_lr5e-5_warm0_wd_0.01_seed788
#nohup python -u train.py --run_name 10/1_$NAME --train_batch_size 32 \
#       --num_train_epochs 10 --learning_rate 5e-5 --warmup_steps 0 --weight_decay 0.01 \
#       --output_dir ./results/$NAME --random_seed 788

#NAME=bert-base_k8:2_lr5e-5_warm0.6_wd_0.01
#nohup python -u train.py --run_name 9/30_$NAME --train_batch_size 32 \
#       --num_train_epochs 10 --learning_rate 5e-5 --warmup_steps 4878 --weight_decay 0.01 \
#       --output_dir ./results/$NAME#/bin/bash
  
