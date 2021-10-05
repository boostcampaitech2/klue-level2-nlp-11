#/bin/bash

NAME=bert-base_lr5e_warm1000_wd0.01_batch64_CE_seed722
python train.py --model_name pre_klue/bert-base --run_name 10/4_$NAME --train_batch_size 64 --eval_batch_size 32 \
        --num_train_epochs 10 --eval_steps 500 --save_steps 500 --learning_rate 5e-5 --warmup_steps 2000 \
       	--weight_decay 0.01 --opt_loss f1 --early_stopping false \
        --output_dir ./results/$NAME --random_seed 722 --save_limit 1\
	--aug aeda --opt_loss CE

#NAME=bert-base_lr5e_warm1000_wd0.01_batch64_CE_seed722
#python train.py --model_name pre_klue/bert-base --run_name 10/4_$NAME --train_batch_size 64 --eval_batch_size 32 \
#        --num_train_epochs 10 --eval_steps 500 --save_steps 500 --learning_rate 5e-5 --warmup_steps 2000 \
#       	--weight_decay 0.01 --opt_loss f1 --early_stopping false \

#       	--weight_decay 0.01 --opt_loss f1 --early_stopping false \
#       	--weight_decay 0.01 --opt_loss f1 --early_stopping false \
#       	--weight_decay 0.01 --opt_loss f1 --early_stopping false \
