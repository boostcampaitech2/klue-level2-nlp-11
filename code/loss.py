import torch
import torch.nn as nn
import torch.nn.functional as F

#focal loss code : https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
  def __init__(self, weight=None, 
                gamma=2., reduction='none'):
    nn.Module.__init__(self)
    self.weight = weight
    self.gamma = gamma
    self.reduction = reduction
        
  def forward(self, input_tensor, target_tensor):
    log_prob = F.log_softmax(input_tensor, dim=-1)
    prob = torch.exp(log_prob)
    return F.nll_loss(
        ((1 - prob) ** self.gamma) * log_prob, 
        target_tensor, 
        weight=self.weight,
        reduction = self.reduction
    )

# f1 loss code : https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1_Loss(nn.Module):
  def __init__(self, epsilon=1e-7):
    super().__init__()
    self.epsilon = epsilon
    self.classes = 30
        
  def forward(self, y_pred, y_true):
    assert y_pred.ndim == 2
    assert y_true.ndim == 1
        
    y_true = F.one_hot(y_true, self.classes).to(torch.float32)
    y_pred = F.softmax(y_pred, dim=1)
        
    tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

    precision = tp / (tp + fp + self.epsilon)
    recall = tp / (tp + fn + self.epsilon)

    f1 = 2 * (precision*recall) / (precision + recall + self.epsilon)
    f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
    return 1 - f1.mean()
