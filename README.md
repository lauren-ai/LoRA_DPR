## Dependencies
Please refer to [PyTorch Homepage](https://pytorch.org/) to install a pytorch version suitable for your system.

Dependencies can be installed by running codes below. Specifically, we use transformers=4.17.0 for our experiments. Other versions should also work well.
```bash
apt-get install parallel
pip install transformers==4.17.0 datasets nltk tensorboard pandas tabulate
```

We use [Tevatron](https://github.com/texttron/tevatron) toolkit for finetuning. You can install it by following its guidelines.

## Pre-training
### Data processing
[MS-Marco documents](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz) are used as unsupervised pre-training corpus. You can download and process the data by running [get_pretraining_data.sh](./get_pretraining_data.sh) script. The processed contextual texts will be stored in **data/msmarco-docs.mlen128.json** as json format.
```bash
bash get_pretraining_data.sh
```

For MS-Marco data processing of Tevatron toolkit, just running script at [msmarco/get_data.sh](msmarco/get_data.sh). The processed data will be stored in **msmarco/marco** folder.
```bash
cd msmarco
bash get_data.sh
