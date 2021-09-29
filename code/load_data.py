import pickle as pickle
import os
import pandas as pd
import torch
from typing import Tuple, List
from torch.utils.data import Dataset, Subset, random_split

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

class RE_Dataset_Default(torch.utils.data.Dataset):
  """ Default Dataset으로 split_dataset 함수를 통해 self.val_ratio 값을 기준으로 train dataset과 val dataset을 분리 """
  def __init__(self, pair_dataset, labels, val_ratio = 0.2):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.val_ratio = val_ratio

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)
  
  def split_dataset(self) -> Tuple[Subset, Subset]:
    n_val = int(len(self) * self.val_ratio)
    n_train = len(self) - n_val
    train_set, val_set = random_split(self, [n_train, n_val])
    return train_set, val_set

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    """ object entity가 올바르지 않게 처리되는 것을 방지하기 위해 코드 수정 """
    i = i[i.find('word')+8: i.find('start_idx')-4]
    j = j[j.find('word')+8: j.find('start_idx')-4]
    
    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences
