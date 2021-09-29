from transformers import AutoConfig, AutoModelForSequenceClassification
from importlib import import_module
import sys

class Model():
    """
    Get pretrained_model from HugginFace
    Get custom_model from ./model
    'custom_model' 파라미터 필요
    ...
    Attributes
    -----------
    name : str
        pre/custom_modelName
    params : dict
        custom_model param (default=None)
        # to be implemented
    Methods
    --------
    get_model(): -> Model
        The method for getting model


    """
    def __init__(self,name, params=None):
        self.classifier = name.split('_')[0]
        self.MODEL_NAME = name.split('_')[1]
        self.params = params
        # self.get_model()

    def get_model(self):
        if self.classifier == 'pre':
            model_config = AutoConfig.from_pretrained(self.MODEL_NAME)
            model_config.num_labels = 30
            model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME, config=model_config)
            return model
        elif self.classifier == 'custom':
            sys.path.append("./model")
            model_module = getattr(import_module(self.MODEL_NAME), self.MODEL_NAME)
            model = model_module(self.params)
            return model
        else:
            print("잘못된 이름 또는 없는 모델입니다.")

if __name__ == '__main__':
    model = Model("custom_testmodel", {"layer":30, "classNum":20}).get_model()
    # print(model.config)