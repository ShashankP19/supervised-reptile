# supervised-reptile with regularization

[Reptile](https://arxiv.org/abs/1803.02999) training code for [Omniglot](https://github.com/brendenlake/omniglot) and [Mini-ImageNet](https://openreview.net/pdf?id=rJY0-Kcll).

Reptile is a meta-learning algorithm that finds a good initialization. It works by sampling a task, training on the sampled task, and then updating the initialization towards the new weights for the task.

This repository contains the Reptile code with regularization added.

# Getting the data

The [fetch_data.sh](fetch_data.sh) script creates a `data/` directory and downloads Omniglot and Mini-ImageNet into it. The data is on the order of 5GB, so the download takes 10-20 minutes on a reasonably fast internet connection.

```
$ ./fetch_data.sh
Fetching omniglot/images_background ...
Extracting omniglot/images_background ...
Fetching omniglot/images_evaluation ...
Extracting omniglot/images_evaluation ...
Fetching Mini-ImageNet train set ...
Fetching wnid: n01532829
Fetching wnid: n01558993
Fetching wnid: n01704323
Fetching wnid: n01749939
...
```

If you want to download Omniglot but not Mini-ImageNet, you can simply kill the script after it starts downloading Mini-ImageNet. The script automatically deletes partially-downloaded data when it is killed early.

# Reproducing training runs

You can train models with the `run_omniglot.py` and `run_miniimagenet.py` scripts. Hyper-parameters are specified as flags (see `--help` for a detailed list). Here are the commands used for the paper:

```shell

# 5-shot 5-way Omniglot.
python -u run_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 5 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 10000 --eval-batch 5 --eval-iters 50 --eval-samples 1000 --checkpoint_org ckpt_o55 --checkpoint_reg ckpt_reg_o55 --org --reg

# 1-shot 5-way Omniglot.
python -u run_omniglot.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 10000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --eval-samples 1000 --checkpoint_org ckpt_o15 --checkpoint_reg ckpt_reg_o15 --org --reg

# 5-shot 5-way Mini-ImageNet.
python -u run_miniimagenet.py --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 10000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --eval-samples 1000 --checkpoint_org ckpt_m55 --checkpoint_reg ckpt_reg_m55 --org --reg

# 1-shot 5-way Mini-ImageNet.
python -u run_miniimagenet.py --shots 1 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 10000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --eval-samples 1000 --checkpoint_org ckpt_m15 --checkpoint_reg ckpt_reg_m15 --org --reg

```

`checkpoint_org` stores checkpoint of model trained using original Reptile code. `checkpoint_reg` stores checkpoint of model trained using Reptile code with regularization.

Use `--org` to run the original Reptile code. Use `--reg` to run the Reptile code with regularization.

Training creates checkpoints. Currently, you cannot resume training from a checkpoint, but you can re-run evaluation from a checkpoint by passing `--pretrained`. You can use TensorBoard on the checkpoint directories to see approximate learning curves during training and testing.

To evaluate with transduction, pass the `--transductive` flag. In this implementation, transductive evaluation is faster than non-transductive evaluation since it makes better use of batches.

