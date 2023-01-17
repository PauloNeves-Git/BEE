# BEE
The code for the BERT Enriched Embedding model, for yield prediction.

### Install
	git clone https://github.com/PauloNeves-Git/BEE.git
	cd BEE
	conda env create -f bee38.yml
	pip install --no-cache torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

### Reproducibility
This code relates to a pre-print available on Chemarxiv https://chemrxiv.org/engage/chemrxiv/article-details/633c44a0ea6a224e2408b7cd. \
The paper discusses the dramatic impact that yield prediction models can already have in the Pharmaceutical Industry and how to improve them, for this reason most results presented in the publication requiere access to industry-scale datasets which are private. 
For the sake of reproducibility a notebook called "BERT Enriched Embedding w Open-Source Data" was created to reproduce the results with open source data. \
\
More notebooks which will enable open-source reproduction of others experiments (uncertainty estimation plots) on the paper will be added.

#### The reaction transformers package in this repository is a combination of modified code from:
- [ ] https://github.com/rxn4chemistry/rxnfp
- [ ] https://github.com/ThilinaRajapakse/simpletransformers
- [ ] https://github.com/huggingface


