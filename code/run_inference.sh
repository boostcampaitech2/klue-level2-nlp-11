#/bin/bash

python inference.py --model_name klue/bert-base \
	--model_dir ./results/bert-base_k8:2_lr5e-5_warm0_wd_0.01_seed452_3/checkpoint-500 \
	--test_csv_path ../dataset/test/test_data.csv --save_path ./prediction/test.csv

