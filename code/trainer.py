import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainerState, TrainerControl, TrainerCallback, TrainingArguments
from loss import *

class CustomTrainer(Trainer):
  def __init__(self, loss_name, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.loss_name = loss_name

  def compute_loss(self, model, inputs, return_outputs=False):
    # config에 저장된 loss_name에 따라 다른 loss 계산
    if self.loss_name == 'f1':
      custom_loss = F1_Loss()
    elif self.loss_name == 'focal':
      custom_loss = FocalLoss()
    elif self.loss_name == 'CE':
      custom_loss = torch.nn.CrossEntropyLoss()
                  
    labels = inputs.pop("labels")
    outputs = model(**inputs)

    if labels is not None:
      loss = custom_loss(outputs.get('logits'), labels)
      if self.loss_name == 'focal':
        loss = loss.mean()
    else:
      loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    
    return (loss, outputs) if return_outputs else loss
