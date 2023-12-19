# InterpretableKGC

## How to train basic KGC model

The related codes are saved in kgc_model/train.py

```
cd /mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/run_scripts

bash train_v0.sh
```

The checkpoint will be saved in kgc_model/saved_checkpoint

The log information will be saved in kgc_model/wandb

In this work, we randomly select a knowledge sentence from the knowledge list corresponding for a wizard turn's generation or prediction.