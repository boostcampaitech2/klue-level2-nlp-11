import pickle as pickle
import os
import pandas as pd
import torch
from typing import Tuple, List
from torch.utils.data import Dataset, Subset, random_split
import random

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


def Data_Sep_Ind(dataset, num):
    '''
    dataset과 몇 덩이로 받을지 num을 입력으로 받는다.
    라벨의 수를 균등하게, 인덱스-리스트의 리스트 형태로 반환한다.
    반환 값 예시 : [[1,4],[0,2],[3,5]]
    '''
    rt = [[] for i in range(num)]
    countdic = {}
    shuffle_ind = [i for i in range(len(dataset))]
    random.shuffle(shuffle_ind)
    for i in shuffle_ind:
        row_data = dataset.loc[i]
        lb = row_data['label']
        countdic[lb] = countdic.get(lb,0)+1
        rt[countdic[lb]%num].append(i)
    return rt

def Dataset_Sep(dataset,fold_num):
    '''
    pandas dataframe과 k-fold의 k를 받는다.
    라벨의 수를 균등하게, Data_SEP_IND를 활용한다.
    k개의 인덱스 뭉치에 대한 데이터프레임을 활용한다. 이를 0~k-1번 데이터프레임이라 하자.
    i번째 dataframe을 dev_data, train_data를 반환하는 제너레이터.
    '''
    N = Data_Sep_Ind(dataset,fold_num)
    df = [dataset.loc[l] for l in N]
    for i in range(fold_num):
        dev_data = df[i]
        train_data = pd.concat([df[j] for j in range(fold_num) if j!=i])
        yield dev_data, train_data
        
    
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
  out_dataset = pd.DataFrame({'id':dataset['id'],
                              'sentence':dataset['sentence'],
                              'subject_entity':subject_entity,
                              'object_entity':object_entity,
                              'label':dataset['label'],
                             })
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
      return_token_type_ids=False
      )
  return tokenized_sentences

#------------------------------------------- entity concat 방식 변경
def custom_tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    temp = f'이 문장에서 {e01}과 {e02}은 어떤 관계일까?'
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      return_token_type_ids=False
      )
  return tokenized_sentences


#------------------------------------------- for typed entity marker ( punct )
def typed_preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  subject_type = []
  object_entity = []
  object_type = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    """ object entity가 올바르지 않게 처리되는 것을 방지하기 위해 코드 수정 """
    se = i[i.find('word')+8: i.find('start_idx')-4]
    st = i[i.find('type')+8: i.find('}')-1]
    
    oe = j[j.find('word')+8: j.find('start_idx')-4]
    ot = j[j.find('type')+8: j.find('}')-1]
    
    subject_entity.append(se)
    subject_type.append(st)
    object_entity.append(oe)
    object_type.append(ot)
  out_dataset = pd.DataFrame({'id':dataset['id'],
                              'sentence':dataset['sentence'],
                              'subject_entity':subject_entity,
                              'subject_type':subject_type,
                              'object_entity':object_entity,
                              'object_type':object_type,
                              'label':dataset['label'],
                             })
  return out_dataset

def typed_load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dt = typed_preprocessing_dataset(pd_dataset)

  # typed entity marker ( punct )
  mf_sen = dt['sentence'].copy()
  se = dt['subject_entity']
  oe = dt['object_entity']
  st = dt['subject_type']
  ot = dt['object_type']
  
  for i in range(len(dt)):
    tmp = mf_sen[i]
    tmp = tmp.replace(se[i], f" ^ * {st[i]} * {se[i]} ^ ")
    tmp = tmp.replace(oe[i], f" # @ {ot[i]} @ {oe[i]} # ")
    tmp = tmp.strip()
    tmp = tmp.replace('  ', ' ')
    mf_sen[i] = tmp
  
  dt['sentence'] = mf_sen.copy()
  return dt

#-------------------
def added_typed_load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dt = typed_preprocessing_dataset(pd_dataset)

  # typed entity marker ( punct )
  mf_sen = dt['sentence'].copy()
  se = dt['subject_entity']
  oe = dt['object_entity']
  st = dt['subject_type']
  ot = dt['object_type']
  
  for i in range(len(dt)):
    tmp = mf_sen[i]
    tmp = tmp.replace(oe[i], f" ^ * {ot[i]} * {oe[i]} ^ ")
    tmp = tmp.replace(se[i], f" # @ {st[i]} @ {se[i]} # ")
    tmp = tmp.strip()
    tmp = tmp.replace('  ', ' ')
    mf_sen[i] = tmp
  
  dt['sentence'] = mf_sen.copy()
  return dt

#-----------------------------------data 추가
def additional_data(data_path, marker):
  config = None
  if marker:
    config = {
      "change_entity": {
          "subject_entity": "object_entity",
          "object_entity": "subject_entity",
          "subject_type": "object_type",
          "object_type": "subject_type",
      },
      "remain_label_list": [
          "per:children",
          "per:other_family",
          "per:colleagues",
          "per:siblings",
          "per:spouse",
          "per:parents",
      ],
      "change_values": {
          "per:parents": "per:children",
          "per:children": "per:parents",
      },
      "cols": ["id", "sentence", "subject_entity", "subject_type", "object_entity", "object_type","label"],
    }
  else:
    config = {
      "change_entity": {
          "subject_entity": "object_entity",
          "object_entity": "subject_entity",
      },
      "remain_label_list": [
          "per:children",
          "per:other_family",
          "per:colleagues",
          "per:siblings",
          "per:spouse",
          "per:parents",
      ],
      "change_values": {
          "per:parents": "per:children",
          "per:children": "per:parents",
      },
      "cols": ["id", "sentence", "subject_entity", "object_entity", "label"],
    }

  # 훈련 데이터를 불러오고 subject_entity와 object_entity만 바꾼다.
  add_data = None
  if marker:
    add_data = added_typed_load_data(data_path).rename(columns=config["change_entity"])
  else:
    add_data = load_data(data_path).rename(columns=config["change_entity"])
  # 추가 데이터를 만들 수 있는 라벨만 남긴다
  add_data = add_data[add_data.label.isin(config["remain_label_list"])]
  # 속성 정렬을 해준다 (정렬을 안할경우 obj와 sub의 순서가 바뀌어 보기 불편함)
  add_data = add_data[config["cols"]]
  # 서로 반대되는 뜻을 가진 라벨을 바꿔준다.
  add_data = add_data.replace({"label": config["change_values"]})
  return add_data

def data_with_addition(data_path, marker_apply):
  if marker_apply:
    added_data = typed_load_data(data_path).append(additional_data(data_path, marker_apply))
  else:
    added_data = load_data(data_path).append(additional_data(data_path, marker_apply))
  added_data.index = [i for i in range(len(added_data))]
  return added_data