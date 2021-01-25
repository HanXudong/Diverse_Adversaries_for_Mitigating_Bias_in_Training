# Diverse_Adversaries_for_Mitigating_Bias_in_Training
Source codes for EACL 2021 paper "Diverse Adversaries for Mitigating Bias in Training"

Xudong Han, Timothy Baldwin and Trevor Cohn (to appear) Diverse Adversaries for Mitigating Bias in Training, In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2021), virtual.

## Environment
```
    python 3.7
    torch
    numpy
    scipy
    scikit-learn
    tqdm
    jupyter notebook
```
## Data
To get the dataset, please follow the instruction from https://github.com/shauli-ravfogel/nullspace_projection
1. Download deepmoji data
    ```console
    mkdir -p data/deepmoji
    wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/pos_pos.npy -P data/deepmoji
    wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/pos_neg.npy -P data/deepmoji
    wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/neg_pos.npy -P data/deepmoji
    wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/neg_neg.npy -P data/deepmoji
    ```
    See [download_data.sh](https://github.com/shauli-ravfogel/nullspace_projection/blob/master/download_data.sh) for more details.
2. Get train, dev, and test splits. `$INPUT_DIR` is where the downloaded files are saved.
   ```
    python deepmoji_split.py \
            --input_dir $INPUT_DIR \
            --output_dir $OUTPUT_DIR
   ```
    Find the `deepmoji_split.py` file from [the INLP repo](https://github.com/shauli-ravfogel/nullspace_projection/blob/master/src/data/deepmoji_split.py).

## Notebooks
- Standard Model: `nb_deepmoji.ipynb`
- INLP Debiasing: `nb_INLP.ipynb`
- Adversarial Debiasing: `nb_adv.ipynb`
- Adversarial Ensemble: `nb_adv_ensemble.ipynb`
- Adversarial Diverse: `nb_differentiated_adv.ipynb`

Notice that in our paper, `Table 1` results are averaged over 10 runs. Above notebooks are just one run rather than averaged scores.
## Experiments
`$data_path` the the folder of splits.

1. Standard model
```console
python script_deepmoji.py \
        --data_path $data_path
```

2. Adv model
```console
python script_deepmoji.py \
        --data_path $data_path \
        --adv \
        --LAMBDA 0.8 \
        --n_discriminator 1
```

3. Ensemble model with $N sub-models
```console
python script_deepmoji.py \
        --data_path $data_path \
        --adv \
        --LAMBDA 0.8 \
        --n_discriminator $N
```

4. Separation model
```console
python script_deepmoji.py \
    --data_path $data_path \
    --adv \
    --LAMBDA 0.8 \
    --n_discriminator $N \
    --DL \
    --diff_LAMBDA 10000
```
