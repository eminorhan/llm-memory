The SLURM scripts in this directory can be used to reproduce the experiments reported in the paper.

Brief description of the scripts:

* Scripts strating with `train_` can be used to train (or finetune) the models on the study set (there is a separate file for each model). 

* Scripts strating with `test_` can be used to run a recognition test on a model (there is a separate file for each model). 

* Scripts strating with `generate_` can be used to run a recall test on a model (again there is a separate file for each model). 

* The script `train_gptj_scratch.sh` can be used to train a `gpt-j-6B` model on the `wikitext-103-raw-v1` dataset from sratch (this model is called `gpt-j-6B-st` in the paper).

* The scripts starting with `retrain_`, `retest_` and `regenerate_` can be used to replicate the retention experiments (training on CNN/Daily Mail corpus, running a recognition test, and running a recall test during the retention phase, respectively). Note that the retention experiments were conducted with the `gpt-j-6B` and `gpt-j-6B-st` models only.