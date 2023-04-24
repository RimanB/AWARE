# AWARE: Contextualizing protein representations using deep learning on interactomes and single-cell experiments

**Authors**:
- [Michelle M. Li](http://michellemli.com)
- [Yepeng Huang](http://zitniklab.hms.harvard.edu)
- [Marissa Sumathipala](http://zitniklab.hms.harvard.edu)
- [Man Qing Liang](http://zitniklab.hms.harvard.edu)
- [Alberto Valdeolivas]()
- [Katherine Liao]()
- [Daniel Marbach]()
- [Marinka Zitnik](http://zitniklab.hms.harvard.edu)

## Overview of AWARE

Protein interaction networks are a critical component in studying the function and therapeutic potential of proteins. However, accurately modeling protein interactions across diverse biological contexts, such as tissues and cell types, remains a significant challenge for existing algorithms.

We introduce AWARE, a flexible geometric deep learning approach that trains on contextualized protein interaction networks to generate context-aware protein representations. Leveraging a multi-organ single-cell transcriptomic atlas of humans, AWARE provides 394,760 protein representations split across 156 cell-type contexts from 24 tissues and organs. We demonstrate that AWARE's contextualized representations of proteins reflect cellular and tissue organization and AWARE's tissue representations enable zero-shot retrieval of tissue hierarchy. Infused with cellular and tissue organization, our contextualized protein representations can easily be adapted for diverse downstream tasks.

We fine-tune AWARE to study the genomic effects of drugs in multiple cellular contexts and show that our context-aware model significantly outperforms state-of-the-art, yet context-agnostic, models. Enabled by our context-aware modeling of proteins, AWARE is able to nominate promising protein targets and cell-type contexts for further investigation. AWARE exemplifies and empowers the long-standing paradigm of incorporating context-specific effects for studying biological systems, especially the impact of disease and therapeutics.

### The AWARE Algorithm

AWARE is a self-supervised geometric deep learning model that can generate protein representations in diverse celltype contexts. It is trained on a set of context-aware protein interaction networks unified by a cellular and tissue network to produce contextualized protein representations based cell type activation. Unlike existing approaches, which do not consider biological context, AWARE produces multiple representations of proteins based on their cell type context, representations of the cell type contexts themselves, and representations of the tissue hierarchy. 

Given the multi-scale nature of the model inputs, AWARE is equipped to learn the topology of proteins, cell types, and tissues in a single unified embedding space. AWARE uses protein-, cell type-, and tissue-level attention mechanisms and objective functions to inject cellular and tissue organization into the embedding space. Intuitively, pairs of nodes that share an edge should be embedded nearby, proteins of the same cell type context should be embedded nearby (and far from proteins in other cell type contexts), and proteins should be embedded close to their cell type context (and far from other cell type contexts).

<p align="center">
<img src="img/aware_overview.png?raw=true" width="700" >
</p>


## Installation and Setup

### :one: Download the Repo

First, clone the GitHub repository:

```
git clone https://github.com/mims-harvard/AWARE
cd AWARE
```

### :two: Set Up Environment

This codebase leverages Python, Pytorch, Pytorch Geometric, etc. To create an environment with all of the required packages, please ensure that [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed and then execute the commands:

```
conda env create -f environment.yml
conda activate aware
bash install_pyg.sh
```

### :three: Download Datasets

The data is hosted on [Figshare](https://figshare.com/articles/software/AWARE). To maintain the directory structure while downloading the files, make sure to select all files and download in the original format. Make sure to also unzip all files in the download.

We provide the following datasets for training AWARE:
- Global reference protein interaction network
- Cell type specific protein interaction networks
- Metagraph of cell type and tissue relationships

The networks are provided in the appropriate format for AWARE. If you would like to use your own set of contextualized networks, please adhere to the format used in the cell type specific protein interaction networks (see [README](https://github.com/mims-harvard/AWARE/blob/main/data_prep/README.md) in `data_prep` folder for more details). The file should be structured as a tab-delimited table, where each line contains information for a single context. Each line must contain the following elements (in this order): index, context name (e.g., cell type name), comma-delimited list of nodes. The lists of nodes are used to extract a subgraph from the global reference network (e.g., global reference protein interaction network).

### :four: Set Configuration File

Go to `project_config.py` and set the project directory (`PROJECT_DIR`) to be the path to the data folder downloaded in the previous step.

If you would like to use your own data, be sure to
1. Modify the data variables in `project_config.py` in lines 10-16.
2. Generate the required shortest path length data files for your patients using the code and instructions in `data_prep/shortest_paths`


### :five: (Optional) Download Model Checkpoints
We also provide checkpoints for AWARE after pretraining. The checkpoints for AWARE can be found [here](https://figshare.com/articles/software/AWARE). You'll need to move them to the directory specified by `project_config.PROJECT_DIR / 'checkpoints'` (see above step). Make sure all downloaded files are unzipped. You can use these checkpoints directly with the scripts in the `finetune_aware` folder instead of training the models yourself.

## Usage

### Finetune AWARE on Your Own Datasets

You can finetune AWARE on your own datasets by using our provided model checkpoints or contextualized representations (i.e., no re-training needed). Please review this [README](https://github.com/mims-harvard/AWARE/blob/main/Finetune-README.md) to learn how to preprocess and finetune AWARE on your own datasets.

### Train AWARE

You can reproduce our results or pretrain AWARE on your own networks:
```
cd aware
python train.py \
        --G_f ppi_edgelist.txt \
        --ppi_f contextual_ppi.csv \
        --cci_bto_f mg_edgelist.txt \
        --save_prefix checkpoints/
```

To see and/or modify the default hyperparameters, please see the `get_hparams()` function in `aware/parse_args.py`.

An example bash script is provided in `aware/run_aware.sh`.

### Visualize AWARE Representations

After training AWARE, you can visualize AWARE's representations using `evaluate/visualize_representations.py`.

### Finetune AWARE for nominating therapeutic targets

After training AWARE (you may also simply use our already-trained models), you can finetune AWARE for any downstream biomedical task of interest. Here, we provide instructions for nominating therapeutic targets. An example bash script can be found [here](https://github.com/mims-harvard/AWARE/blob/main/finetune_aware/run_model.sh).

The results of the `train.py` script are found in 
```
project_config.PROJECT_RESULTS/<TASK>/<DATASET_NAME>
```
where
- `<TASK>` is `tx_target`
- `<DATASET_NAME>` is the name of the therapeutic area

:sparkles: To finetune AWARE for nominating therapeutic targets of rheumatoid arthritis:

```
cd shepherd
python predict.py \
        --task tx_target \
        --disease EFO_0000685 \
        --embeddings_dir checkpoints/
```

:sparkles: To finetune AWARE for nominating therapeutic targets of inflammatory bowel disease:

```
cd shepherd
python predict.py \
        --task tx_target \
        --disease EFO_0003767 \
        --embeddings_dir checkpoints/
```

To generate predictions on a different therapeutic area, simply find the disease ID from OpenTargets and change the `---disease` flag.

To see and/or modify the default hyperparameters, please see the `get_hparams()` function in `finetune_aware/train_utils.py`.

## Additional Resources

- [Paper](h)
- [Project Website](https://zitniklab.hms.harvard.edu/projects/AWARE/)

```
@article{aware,
  title={Contextualizing protein representations using deep learning on interactomes and single-cell experiments},
  author={Li, Michelle M. and Huang, Yepeng and Sumathipala, Marissa and Liang, Man Qing Valdeolivas, Alberto and Liao, Katherine and Marbach, Daniel and Zitnik, Marinka},
  journal={bioRxiv},
  year={2023}
}
```


## Questions

Please leave a Github issue or contact Michelle Li at michelleli@g.harvard.edu.
