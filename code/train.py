import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, EarlyStoppingCallback, TrainerState, TrainerControl, TrainerCallback, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, set_seed
from load_data import *
import wandb
import argparse
import models
import random
from custom_callback import MyCallback
from custom_early_stopping import MyEarlyStoppingCallback

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)
      'no_relation' : 관계가 존재하지 않는다.
      'org:dissolved' : 지정된 조직이 해산한 날짜
      'org:founded' : 지정된 조직이 설립된 날짜
      'org:place_of_headquarters' : 조직의 본사가 위치한 장소
      'org:alternate_names' : 지정된 조직의 별칭
      'org:member_of' : 지정된 조직에 속하는 조직
      'org:members' : 지정된 조직에 속하는 조직들
      'org:political/religious_affiliation' : 지정된 조직이 속한 정치/종교 그룹
      'org:product' : 지정된 조직에 의해 생산된 상품들
      'org:founded_by' : 지정된 조직을 설립한 사람이나 조직
      'org:top_members/employees' : 지정된 조직의 대표 또는 구성원
      'org:number_of_employees/members' : 지정 조직에 소속된 총 구성원 수
      'per:date_of_birth' : 지정된 사람이 태어난 날짜
      'per:date_of_death' : 지정된 사람이 죽은 날짜
      'per:place_of_birth' : 지정된 사람이 태어난 장소
      'per:place_of_death' : 지정된 사람이 죽은 장소
      'per:place_of_residence' : 지정된 사람이 거주하는 장소
      'per:origin' : 지정된 사람의 출신이나 국적
      'per:employee_of' : 지정된 사람이 일하는 조직
      'per:schools_attended' : 지정된 사람이 다녔던 학교
      'per:alternate_names' : 지정된 사람의 별명
      'per:parents' : 지정된 사람의 부모
      'per:children : 지정된 사람의 자식
      'per:siblings' : 지정된 사람의 형제자매
      'per:spouse' : 지정된 사람의 배우자
      'per:other_family' : 부모, 자식 형제자매, 배우자를 제외한 지정된 사람의 가족 구성원
      'per:colleagues' : 지정된 사람과 함께 일하는 동료
      'per:product' : 지정된 사람이 제작한 제품이나 미술품
      'per:religion' : 지정된 사람이 믿는 종교
      'per:title' : 지정된 사람의 직업적 지위를 나타내는 공식 또는 비공식 이름
    """
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label, dict_pkl):
  num_label = []
  with open(dict_pkl, 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train(args):
  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"

  args_model_name = args.model_name
  classfier = args_model_name.split("_")[0]
  MODEL_NAME = args_model_name.split("_")[1]
  fold_k_num = args.k_num
  iter_num = args.iter_num

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  set_seed(args.random_seed)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)
  params = None
  if classfier == "custom":
    ### args 로 parameter들 받기
    params = {"layer":30, "classNum":20} # for testing -> should be implemented

  Load_dataset = load_data("../dataset/train/train.csv")
  for model_num, (dev_dataset, train_dataset) in enumerate(Dataset_Sep(Load_dataset,fold_k_num)):
    if model_num == iter_num:
        break

    if torch.cuda.is_available():
      print("="*40)
      torch.cuda.empty_cache()
      print("cuda empty cache!!")
      print("="*40)

    train_label = label_to_num(train_dataset['label'].values, args.label_to_num)
    dev_label = label_to_num(dev_dataset['label'].values,args.label_to_num)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    model = models.Model(name=args_model_name, params=params).get_model()
    # print(model.config)
    # model.parameters
    model.to(device)
  
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        save_total_limit=args.save_limit,
        save_steps=args.save_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps = args.eval_steps,
        load_best_model_at_end = True ,
        report_to="wandb",
        run_name=args.run_name,
        metric_for_best_model = "micro f1 score" # -> change criterion for saving best model
        )
    callback_list = []
    if str2bool(args.custom_callback):
        print("="*40)
        callback_list.append(MyCallback)
        print("Custom Callback is applied !!")
    if str2bool(args.early_stopping):
        print("="*40)
        callback_list.append(MyEarlyStoppingCallback(args.early_stopping_patience))
        print("Early Stopping is applied !!")
    print("="*40)
    print(f"callback_list : {callback_list}")
    print("="*40)
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,         # define metrics function
        callbacks=callback_list
    )
    # train model
    trainer.train()
    model.save_pretrained(args.best_model_dir)

def str2bool(bool_str):
    bool_str = bool_str.lower()
    if bool_str in ("yes", "y", "t", "true"):
        return True
    elif bool_str in ("no", "n", "f", "false"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean Value Expected!!")

if __name__ == '__main__':


  wandb.login()
  parser = argparse.ArgumentParser()
  parser.add_argument('--random_seed', type=int, default=211, help='random seed setting')
  parser.add_argument('--k_num', type=int, default=5, help='ratio : validation data(1) and train data(k-1)')
  parser.add_argument('--iter_num', type=int, default=1, help='maximum k_num. get nums of model.')

  parser.add_argument('--model_name', type=str, default='pre_klue/bert-base', help='model name')
  parser.add_argument('--train_csv_path', type=str, default='../dataset/train/train.csv', help='train data csv path')
  parser.add_argument('--label_to_num', type=str, default='dict_label_to_num.pkl', help='dictionary information of label to number')
  parser.add_argument('--num_to_label', type=str, default='dict_num_to_label.pkl', help='dictionary information of number to label')
  parser.add_argument('--num_labels', type=int, default=30, help='number of labels')
  parser.add_argument('--output_dir', type=str, default='./results', help='output directory')
  parser.add_argument('--save_limit', type=int, default=5, help='number of total save model')
  parser.add_argument('--save_steps', type=int, default=500, help='model saving step')
  parser.add_argument('--num_train_epochs', type=int, default=20, help='total number of training epochs')
  parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
  parser.add_argument('--train_batch_size', type=int, default=32, help='batch size per device during training')
  parser.add_argument('--eval_batch_size', type=int, default=32, help='batch size for evaluation')
  parser.add_argument('--warmup_steps', type=int, default=500, help='number of warmup steps for learning rate scheduler')
  parser.add_argument('--weight_decay', type=float, default=0.01, help='strength of weight decay')
  parser.add_argument('--logging_dir', type=str, default='./logs', help='directory for storing logs')
  parser.add_argument('--logging_steps', type=int, default=100, help='log saving step')
  parser.add_argument('--evaluation_strategy', type=str, default='steps', help='evaluation strategy to adopt during training')
  parser.add_argument('--eval_steps', type=int, default=500, help='evaluation step')
  parser.add_argument('--best_model_dir', type=str, default='./best_model', help='best model directory')
  parser.add_argument('--run_name', type=str, default="experiment", help='wandb run name')
  parser.add_argument('--early_stopping', type=str, default="true", help='if true, you can apply EarlyStopping')
  parser.add_argument('--custom_callback', type=str, default="true", help='if true, you can apply CustomCallback')
  parser.add_argument('--early_stopping_patience', type=int, default=3, help='the number of early_stopping_patience')
  args = parser.parse_args()
  random.seed(args.random_seed)

  train(args)

# export WANDB_PROJECT=KLUE
