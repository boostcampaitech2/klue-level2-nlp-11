from transformers import TrainerState, TrainerControl, TrainerCallback, TrainingArguments

class MyCallback(TrainerCallback):
    """A callback that prints a message while training"""

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """학습 시작할때 Log"""
        print("="*40)
        print("Start training!!!")
        print("="*40)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """ 새로운 epoch이 시작될때 log """
        print("="*40)
        print(f"{state.epoch} epoch starts !!!")
        print("="*40)