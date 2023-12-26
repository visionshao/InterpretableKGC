# InterpretableKGC

## How to train basic KGC model

The related codes are saved in kgc_model/train.py

```
cd InterpretableKGC/codes/run_scripts

bash train_v0.sh
```

The checkpoint will be saved in kgc_model/saved_checkpoint

The log information will be saved in kgc_model/wandb

In this work, we randomly select a knowledge sentence from the knowledge list corresponding for a wizard turn's generation or prediction.

## Get generation

```
cd InterpretableKGC/codes/run_scripts

bash gen_response_for_knowledge.sh
```

For train datasets, you could set start index and end index (--train_s and --train_e) to split data and thus speedup generation

The outputs will be saved to InterpretableKGC/codes/gen_response/generation_results

For each json file, it contains a dictionary whose keys includes: dialogue_context, knowledges, response, generation

## No-reference Metric Construction

First enter the no-reference metric codes directory.

```
cd programs/InterpretableKGC/codes/metrics/no_reference
```

Then, train the mlm_score metric model (AutoMaskedLanguageModel) first.

```
cd mlm_score 

python train.py
```
The checkpoints are saved in mlm_score/save_mdoels/.

Next, we should train the dr metric model (AutoSequenceClassificationModel). Before that, we still need to copy lacked files (*.json. *.txt) from programs/InterpretableKGC/codes/metrics/no_reference/mlm_score/pretrain_models/roberta_base to the best checkpoint of mlm_score model. The dr model is based on this best mlm_score model.

```
cd dr

python train.py
```

The checkpoints are saved in mlm_score/save_mdoels/.

## Construction of Huggingface Datasets with No-Reference Scores

```
cd /InterpretableKGC/codes/metrics/no_reference

python score.py
```

This score.py will load datasets from InterpretableKGC/codes/gen_response/generation_results and outputs the datasets with mlm scores and dr scores for generations in the save dir.



