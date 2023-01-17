__all__ = ['logger', 'SmilesLanguageModelingModel', 'SmilesClassificationModel']

import os
import numpy as np
import pandas as pd
import torch
import logging
import random
import warnings
import pkg_resources
import sklearn

from config.configuration_bert import BertConfig
from modeling_bert import BertForMaskedLM

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

from tokenization import SmilesTokenizer
logger = logging.getLogger(__name__)

from config.global_args import global_args
from language_modeling_model import LanguageModelingModel

class SmilesLanguageModelingModel(LanguageModelingModel):
    def __init__(
        self,
        model_type,
        model_name,
        generator_name=None,
        discriminator_name=None,
        train_files=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        **kwargs,
    ):

        """
        Initializes a LanguageModelingModel.
        Main difference to https://github.com/ThilinaRajapakse/simpletransformers/blob/master/simpletransformers/classification/classification_model.py
        is that it uses a SmilesTokenizer instead of the original Tokenizer.
        Args:
            model_type: The type of model bert (other model types could be implemented)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            generator_name (optional): A pretrained model name or path to a directory containing an ELECTRA generator model.
            discriminator_name (optional): A pretrained model name or path to a directory containing an ELECTRA discriminator model.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            train_files (optional): List of files to be used when training the tokenizer.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {"bert": (BertConfig, BertForMaskedLM, SmilesTokenizer)}

        if args and "manual_seed" in args:
            random.seed(args["manual_seed"])
            np.random.seed(args["manual_seed"])
            torch.manual_seed(args["manual_seed"])
            if "n_gpu" in args and args["n_gpu"] > 0:
                torch.cuda.manual_seed_all(args["manual_seed"])

        self.args = {
            "block_size": -1,
            "config_name": None,
            "dataset_class": None,
            "dataset_type": "None",
            "discriminator_config": {},
            "discriminator_loss_weight": 50,
            "generator_config": {},
            "max_steps": -1,
            "min_frequency": 2,
            "mlm": True,
            "mlm_probability": 0.15,
            "sliding_window": False,
            "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            "stride": 0.8,
            "tie_generator_and_discriminator_embeddings": True,
            "tokenizer_name": None,
            "vocab_size": None,
            "local_rank": -1,
        }


        self.args.update(global_args)

        saved_model_args = self._load_model_args(model_name)
        if saved_model_args:
            self.args.update(saved_model_args)

        if args:
            self.args.update(args)

        if self.args["local_rank"] != -1:
            logger.info(f'local_rank: {self.args["local_rank"]}')
            torch.distributed.init_process_group(backend="nccl")
            cuda_device = self.args["local_rank"]

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        self.results = {}

        if not use_cuda:
            self.args["fp16"] = False

        self.args["model_name"] = model_name
        self.args["model_type"] = model_type

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.tokenizer = tokenizer_class(self.args["vocab_path"])

        self.config = config_class(**self.args["config"], **kwargs)

        self.config.vocab_size = len(self.tokenizer)


        if self.args["block_size"] <= 0:
            self.args["block_size"] = min(self.args["max_seq_length"], self.tokenizer.max_len)
        else:
            self.args["block_size"] = min(self.args["block_size"], self.tokenizer.max_len, self.args["max_seq_length"])

        if self.args["model_name"]:
            self.model = model_class.from_pretrained(
                model_name, config=self.config, cache_dir=self.args["cache_dir"], **kwargs,
            )
        else:
            logger.info(" Training language model from scratch")

            self.model = model_class(config=self.config)
            model_to_resize = self.model.module if hasattr(self.model, "module") else self.model
            model_to_resize.resize_token_embeddings(len(self.tokenizer))

        if model_type in ["camembert", "xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion."
            )
            self.args["use_multiprocessing"] = False

        if self.args["wandb_project"] and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args["wandb_project"] = None


from classification_model import ClassificationModel
from modeling_bert import BertForSequenceClassification


class SmilesClassificationModel(ClassificationModel):
    def __init__(
        self, model_type, model_name, num_labels=None, weight=None, freeze_encoder=False, freeze_all_but_one=False, freeze_for_eqvs=False, args=None, use_cuda=True, cuda_device=-1, **kwargs,
    ):

        """
        Initializes a SmilesClassificationModel model.

        Main difference to https://github.com/ThilinaRajapakse/simpletransformers/blob/master/simpletransformers/classification/classification_model.py
        is that it uses a SmilesTokenizer instead of the original Tokenizer

        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            num_labels (optional): The number of labels or classes in the dataset.
            weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {"bert": (BertConfig, BertForSequenceClassification, SmilesTokenizer)}

        if args and "manual_seed" in args:
            random.seed(args["manual_seed"])
            np.random.seed(args["manual_seed"])
            torch.manual_seed(args["manual_seed"])
            if "n_gpu" in args and args["n_gpu"] > 0:
                torch.cuda.manual_seed_all(args["manual_seed"])

        self.args = {
            "sliding_window": False,
            "tie_value": 1,
            "stride": 0.8,
            "regression": False,
            "lazy_text_column": 0,
            "lazy_text_a_column": None,
            "lazy_text_b_column": None,
            "lazy_labels_column": 1,
            "lazy_header_row": True,
            "lazy_delimiter": "\t",
        }

        self.args.update(global_args)

        saved_model_args = self._load_model_args(model_name)
        if saved_model_args:
            self.args.update(saved_model_args)

        if args:
            self.args.update(args)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if num_labels:
            self.config = config_class.from_pretrained(model_name, num_labels=num_labels, **self.args["config"])
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args["config"])
            self.num_labels = self.config.num_labels
        self.weight = weight

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if self.weight:
            self.model = model_class.from_pretrained(
                model_name, config=self.config, weight=torch.Tensor(self.weight).to(self.device), **kwargs,
            )
        else:
            self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)

        self.results = {}

        if not use_cuda:
            self.args["fp16"] = False

        self.tokenizer = tokenizer_class(os.path.join(model_name, 'vocab.txt'))

        if freeze_encoder:
            for name, param in self.model.named_parameters():
                if 'classifier' in name:
                    continue
                param.requires_grad = False
        elif freeze_all_but_one:
            n_layers = self.model.config.num_hidden_layers
            for name, param in self.model.named_parameters():
                if str(n_layers-1) in name:
                    continue
                elif 'classifier' in name:
                    continue
                elif 'pooler' in name:
                    continue
                param.requires_grad = False
        elif "freeze_for_eqvs" in kwargs.keys() and kwargs["freeze_for_eqvs"]:
            print("freeze_for_eqvs was executed!")
            for name, param in self.model.named_parameters():
                if 'classifier' in name:
                    continue
                elif 'equivalent' in name:
                    continue
                elif "0" in name:
                    continue
                elif 'pooler' in name:
                    continue
                param.requires_grad = False

        self.args["model_name"] = model_name
        self.args["model_type"] = model_type


        if self.args["wandb_project"] and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args["wandb_project"] = None