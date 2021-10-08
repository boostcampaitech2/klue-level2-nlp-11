# KLUE - Relation Extraction
A solution for KLUE Relation Extraction Competition in the 2nd BoostCamp AI Tech by team AI-ESG 

## Content
- [Background - RE(Relation Extraction) Tasks](#background---rerelation-extraction-tasks)
- [Project Outline](#project-outline)
- [Team](#team)
	- [Members of Team AI-ESG](#members-of-team-ai-esg)
- [Structure](#structure)
- [Getting Started](#getting-started)
	- [Hardware](#hardware)
	- [Dependencies](#dependencies)
	- [Install Requirements](#install-requirements)
	- [Train](#train)
	- [Inference](#inference)
	- [Ensemble](#ensemble)
	- [TATP](#tatp)
		- [Generate Text](#generate-text)
		- [Train TATP](#train-tatp)


## Background - RE(Relation Extraction) Tasks

Relation extraction task predicts attributes and relations between entities in sentence

**This is an example:**

***sentence**: **오라클**(구 **썬 마이크로시스템즈**)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
**subject-entity**: 썬 마이크로시스템즈
**object-entity**: 오라클
**relation**: 단체:별칭 (org:alternatenames)*




## Project Outline
* input: sentence, subject_entity, object_entity
* output: pred_label, probs
    * classes (30)
        * 'no_relation': 0
        * 'org:top_members/employees': 1
        * 'org:members': 2
        * 'org:product': 3
        * 'per:title': 4
        * 'org:alternate_names': 5
        * 'per:employee_of': 6
        * 'org:place_of_headquarters': 7
        * 'per:product': 8
        * 'org:number_of_employees/members': 9
        * 'per:children': 10
        * 'per:place_of_residence': 11
        * 'per:alternate_names': 12
        * 'per:other_family': 13
        * 'per:colleagues': 14
        * 'per:origin': 15
        * 'per:siblings': 16
        * 'per:spouse': 17
        * 'org:founded': 18
        * 'org:political/religious_affiliation': 19
        * 'org:member_of': 20
        * 'per:parents': 21
        * 'org:dissolved': 22
        * 'per:schools_attended': 23
        * 'per:date_of_death': 24
        * 'per:date_of_birth': 25
        * 'per:place_of_birth': 26
        * 'per:place_of_death': 27
        * 'org:founded_by': 28
        * 'per:religion': 29

## Team

### Members of Team AI-ESG
| Name | github | contact |
| -------- | -------- | -------- |
| 문석암     | [Link](https://github.com/mon823) | mon823@naver.com |
| 박마루찬 | [Link](https://github.com/MaruchanPark) | shaild098@naver.com |
| 박아멘 | [Link](https://github.com/AmenPark) | puzzlistpam@gmail.com |
| 우원진 | [Link](https://github.com/woowonjin) | dndnjswls613@naver.com |
| 윤영훈 | [Link](https://github.com/wodlxosxos) | wodlxosxos73@gmail.com |
| 장동건 | [Link](https://github.com/mycogno) | jdg4661@gmail.com |
| 홍현승 | [Link](https://github.com/Hong-Hyun-Seung) | honghyunseung100@gmail.com |

## Structure
```
├── code
│   ├── best_model
│   ├── dict_label_to_num.pkl
│   ├── dict_num_to_label.pkl
│   ├── ensemble.py
│   ├── inference.py
│   ├── load_data.py
│   ├── loss.py
│   ├── models.py
│   ├── prediction
│   │   └── sample_submission.csv
│   ├── requirements.txt
│   ├── results
│   ├── trainer.py
│   └── train.py
└── dataset
    ├── test
    │   └── test_data.csv
    └── train
        └── train.csv
```
* ```dataset/``` : train, test datasets, but not contained in this repository
* ```code/best_model/``` : When you train a model, the best scored checkpoint model will be saved here.
* ```code/results/```: saved checkpoints
* ```code/prediction/```: saved inferenced csv files

## Getting Started
### Hardware
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-SXM2-32GB

### Dependencies
- pandas==1.1.5
- scikit-learn~=0.24.1
- transformers==4.10.0
- torch==1.6.0
- wandb==0.12.1

### Install Requirements
```
pip install -r requirements.txt
```

### Train
You can train our model with train.py
It includes various arguments, which can be set.

**This is an example:**
```
python train.py --run_name NAME --train_batch_size 32 \
        --num_train_epochs 10 --learning_rate 5e-5 --warmup_steps 0 --weight_decay 0.01 \
        --output_dir ./results/NAME --random_seed 452
```
or run following shell script file.
```$ ./run_train.sh```

### Inference
```
python inference.py --model_name klue/roberta-large --model_dir ./best_model
```
### Ensemble
```
python ensemble.py --csv_name output1.csv,output2.csv --csv_dir ./prediction \
                --save_path ./prediction/ensemble.csv
```
### TATP

#### Generate Text
**This is an example:**
```
python ./model/mk_text.py
```

#### Train TATP
**This is an example:**
```
python ./model/maskedml_for_tatp.py \
        --model_name_or_path [klue/roberta-large] --run_name [NAME] \
        --do_train --output_dir [model path]
```
