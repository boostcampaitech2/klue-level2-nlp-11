from transformers import TrainerState, TrainerControl, TrainerCallback, TrainingArguments, EarlyStoppingCallback
import numpy as np

class MyEarlyStoppingCallback(EarlyStoppingCallback):
    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):  
            self.compare_metric_val = state.best_metric
            self.early_stopping_patience_counter = 0
        else:
            self.compare_metric_val = metric_value
            self.early_stopping_patience_counter += 1
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        if self.early_stopping_patience_counter == 0:
            print("="*40)
            print("Best Model Improved !!")
            print(f"Model micro_f1_score improved from {self.compare_metric_val} to {state.best_metric}")
            print(f"epoch: {state.epoch},  best_metric[ micro_f1_score ]: {state.best_metric},  best_model_checkpoint: {state.best_model_checkpoint}")
            print("="*40)
        elif self.early_stopping_patience_counter > 0:
            print("="*40)
            print("Best Model isn't improved !!")
            print(f"Model micro_f1_score is not improved from {state.best_metric} to {self.compare_metric_val}")
            print(f"epoch: {state.epoch},  best_metric[ micro_f1_score ]: {state.best_metric},  best_model_checkpoint: {state.best_model_checkpoint}")
            print("="*40)