import os
import os
import shutil

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

class SaveCheckpointCallback(TrainerCallback):
    def on_save(self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs, ):
        if args.local_rank == 0 or args.local_rank == -1:
            # Save the best model
            best_checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-best")

            if state.best_model_checkpoint and os.path.exists(state.best_model_checkpoint):
                if os.path.exists(best_checkpoint_folder):
                    shutil.rmtree(best_checkpoint_folder)
                shutil.copytree(state.best_model_checkpoint, best_checkpoint_folder)

            print("\n" + "="*100)
            print(f"\nLatest saved checkpoint: {os.path.join(args.output_dir, f'{PREFIX_CHECKPOINT_DIR}-{state.global_step}')}")
            print(f"\nBest checkpoint: {state.best_model_checkpoint}\nWER: {state.best_metric}")
            print(f"\nthe reported evaluation loss is meaningless and should be ignored.\n")
            print("="*100 + "\n")
        return control
