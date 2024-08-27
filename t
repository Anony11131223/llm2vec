{'input_ids': tensor([[128009, 128009, 128009,  ...,   1229,     18,   3135],
        [128009, 128009, 128009,  ...,    311,   1948,  21425],
        [128009, 128009, 128009,  ...,   3775,    482,   8336],
        ...,
        [128009, 128009, 128009,  ...,     43,     41,   4977],
        [128009, 128009, 128009,  ...,     79,   2096,     65],
        [128009, 128009, 128009,  ...,  30693,    753,   4333]],
       device='cuda:0'), 'attention_mask': tensor([[0, 0, 0,  ..., 1, 1, 1],
        [0, 0, 0,  ..., 1, 1, 1],
        [0, 0, 0,  ..., 1, 1, 1],
        ...,
        [0, 0, 0,  ..., 1, 1, 1],
        [0, 0, 0,  ..., 1, 1, 1],
        [0, 0, 0,  ..., 1, 1, 1]], device='cuda:0'), 'embed_mask': tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]], device='cuda:0')}


   def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str,Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) ->Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        print("print inputs: ",inputs)
        #labels=inputs.pop("labels")
        input_dict, labels=inputs

        outputs=model(input_dict)#(sentence_feature)#(input_ids)#, embed_mask"=embed_mask)#, attention_mask=attention_mask)#model(**input_dict)
        representation=outputs.last_hidden_state[:, 0, :]
        predictions=self.mlp(representation).squeeze(-1)


  File "/usr/local/lib/python3.10/dist-packages/peft/peft_model.py", line 762, in forward
    return self.get_base_model()(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
TypeError: LlamaModel.forward() got an unexpected keyword argument 'embed_mask'
  0%|                                                     | 0/225 [00:00<?, ?it/s]
        loss=nn.MSELoss()(predictions, labels.float())
        if return_outputs:
            return (loss, outputs)
        return loss

class ReturnPredictionTrainer(Trainer):
    def __init__(
        self,
        *args,
        mlp_hidden_dim:int=4096,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mlp=nn.Sequential(
            nn.Linear(self.model.config.hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mlp.to(device)#(self.model.device)
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str,Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) ->Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        print("print inputs: ",inputs)
        #labels=inputs.pop("labels")
        input_dict, labels=inputs
        embed_mask=input_dict.pop("embed_mask",None)
        #input_ids=input_dict["input_ids"]
        #attention_mask=input_dict.get("attension_mask", None)
        #embed_mask=input_dict.get("embed_mask", None)
        # q_reps=self.model(features[0])
        # d_reps=self.model(features[1])
        # d_reps_neg=None
        # if len(features)>2:
        #   d_reps_neg=self.model(features[2])

        # sentence_feature={
        #   "input_ids": input_dict.get["input_ids"], #q_reps,
        #   "attention_mask": input_dict.get["attension_mask"],
        #   "embed_mask": input_dict.get["embed_mask"]
        # }
        # if d_reps_neg is not None:
        #   sentence_feature["embed_mask"]=d_reps_neg

        outputs=model(input_dict)#(sentence_feature)#(input_ids)#, embed_mask"=embed_mask)#, attention_mask=attention_mask)#model(**input_dict)
        print("outputs:", outputs)
        print(outputs.shape)
        representation=outputs.last_hidden_state[:, 0, :]
        predictions=self.mlp(representation).squeeze(-1)
        loss=nn.MSELoss()(predictions, labels.float())
        if return_outputs:
            return (loss, outputs)
        return loss

    def _save(self, output_dir: Optional[str]=None, state_dict=None):
        output_dir="/content/drive/MyDrive/llm2vec-main2"#output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.model.save_pretrained(output_dir)
        torch.save(self.mlp.state_dict(), os.path.join(output_dir, "mlp.pt"))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

 File "/content/drive/MyDrive/llm2vec-main2/experiments/run_simcse.py", line 378, in compute_loss
    representation=outputs.last_hidden_state[:, 0, :]
AttributeError: 'Tensor' object has no attribute 'last_hidden_state'


 File "/content/drive/MyDrive/llm2vec-main2/experiments/run_simcse.py", line 379, in compute_loss
    predictions=self.mlp(representation).squeeze(-1)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", line 219, in forward
    input = module(input)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/linear.py", line 117, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float




    loss = self.compute_loss(model, inputs)
  File "/content/drive/MyDrive/llm2vec-main2/experiments/run_simcse.py", line 376, in compute_loss
    outputs=model(input_dict)#(sentence_feature)#(input_ids)#, embed_mask"=embed_mask)#, attention_mask=attention_mask)#model(**input_dict)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/content/drive/MyDrive/llm2vec-main2/llm2vec/llm2vec.py", line 238, in forward
    reps = self.model(**sentence_feature)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/peft/peft_model.py", line 762, in forward
    return self.get_base_model()(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py", line 931, in forward
    inputs_embeds = self.embed_tokens(input_ids)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)


import logging
from dataclasses import dataclass, field
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import pandas as pd
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import seed_worker

from peft import LoraConfig, get_peft_model
import os
# 添加环境变量
# os.environ['MY_VARIABLE'] ="/home/rliuaj/llm2vec-main/"

import sys

sys.path.append("/content/drive/MyDrive/llm2vec-main2")


from llm2vec import LLM2Vec
from llm2vec.dataset.utils import load_dataset
from llm2vec.loss.utils import load_loss

from tqdm import tqdm
from datasets import Dataset

transformers.logging.set_verbosity_error()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def load_dataset_from_dataframe(file_path: str):
    df=pd.read_parquet(file_path) #'/home/rliuaj/text/gpt_api/tfns.parquet'
    # dataset=Dataset.from_pandas(df)
    # return dataset
    examples=[{"input":row["input"], "output":row["output"]} for _,row in df.iterrows()]
    return examples


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
        "GemmaConfig",
        "Qwen2Config",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The PEFT model checkpoint to add on top of base model.")},
    )
    bidirectional: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable bidirectional attention in the model. If set to False, the model will use unidirectional attention."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    pooling_mode: Optional[str] = field(
        default="mean",
        metadata={
            "help": ("The pooling mode to use in the model."),
            "choices": ["mean", "weighted_mean", "eos_token"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use. Options: E5"},
    )
    dataset_file_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file or folder."}
    )
    # TODO: implement this
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


@dataclass
class CustomArguments:
    """
    Custom arguments for the script
    """

    simcse_dropout: float = field(
        default=0.1, metadata={"help": "The SimCSE dropout rate for the model"}
    )

    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )

    lora_r: int = field(default=8, metadata={"help": "The r value for lora"})

    stop_after_n_steps: int = field(
        default=10000, metadata={"help": "Stop training after n steps"}
    )

    experiment_id: Optional[str] = field(
        default=None, metadata={"help": "The experiment id"}
    )

    loss_class: Optional[str] = field(
        default="HardNegativeNLLLoss",
        metadata={
            "help": "The loss class to use for training. Options: HardNegativeNLLLoss"
        },
    )

    loss_scale: float = field(
        default=50.0, metadata={"help": "The loss scale for the loss function"}
    )


# @dataclass
# class DefaultCollator:
#     model: LLM2Vec

#     def __init__(self, model: LLM2Vec) -> None:
#         self.model = model

#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#         batch = features
#         num_texts = len(batch[0]['input'])
#         texts = [[] for _ in range(num_texts)]
#         labels = []

#         for example in batch:
#             for idx, text in enumerate(example['input']):
#                 # TODO: Add prepare_for_tokenization here similar to supervised training and see if it impacts performance
#                 texts[idx].append(text)
#             labels.append(example['output'])
#         labels = torch.tensor(labels)

#         sentence_features = []
#         for idx in range(num_texts):
#             tokenized = self.model.tokenize(texts[idx])
#             sentence_features.append(tokenized)

#         return sentence_features, labels

@dataclass
class DefaultCollator:
    model: LLM2Vec

    def __init__(self, model: LLM2Vec) -> None:
        self.model = model

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts=[example['input'] for example in features]
        labels=torch.tensor([example['output'] for example in features], dtype=torch.float)
        
        # sentence_features=[]
        # for text in texts:
        #   tokenized=self.model.tokenize([texts])
          # sentence_features.append(tokenized)
        #tokenized=self.model.tokenize(texts)
        # sentence_features = []
        # for idx in range(len(labels)):
        #     tokenized = self.model.tokenize(texts[idx])
        #     sentence_features.append(tokenized)
        tokenized_features = self.model.tokenize(texts)
        return tokenized_features, labels
        #{"input_ids":tokenized["input_ids"],"attention_mask":tokenized["attention_mask"],
         # "labels":labels}#sentence_features, labels


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


# class SimCSETrainer(Trainer):

#     def __init__(
#         self,
#         *args,
#         loss_function=None,
#         **kwargs,
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         self.loss_function = loss_function

#     def compute_loss(
#         self,
#         model: nn.Module,
#         inputs: Dict[str, Union[torch.Tensor, Any]],
#         return_outputs: bool = False,
#     ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
#         features, labels = inputs
#         q_reps = self.model(features[0])
#         d_reps = self.model(features[1])

#         d_reps_neg = None
#         if len(features) > 2:
#             d_reps_neg = self.model(features[2])

#         loss = self.loss_function(q_reps, d_reps, d_reps_neg)

#         if return_outputs:
#             output = torch.cat(
#                 [model(row)["sentence_embedding"][:, None] for row in features], dim=1
#             )
#             return loss, output

#         return loss

#     def _save(self, output_dir: Optional[str] = None, state_dict=None):
#         # If we are executing this function, we are the process zero, so we don't check for that.
#         output_dir = output_dir if output_dir is not None else self.args.output_dir
#         os.makedirs(output_dir, exist_ok=True)
#         logger.info(f"Saving model checkpoint to {output_dir}")

#         self.model.save(output_dir)

#         # Good practice: save your training arguments together with the trained model
#         torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

class ReturnPredictionTrainer(Trainer):
    def __init__(
        self,
        *args,
        mlp_hidden_dim:int=4096,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mlp=nn.Sequential(
            nn.Linear(self.model.config.hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mlp.to(device)#(self.model.device)
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str,Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) ->Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        print("print inputs: ",inputs)
        #labels=inputs.pop("labels")
        input_dict, labels=inputs
        embed_mask=input_dict.pop("embed_mask",None)
        #input_ids=input_dict["input_ids"]
        #attention_mask=input_dict.get("attension_mask", None)
        #embed_mask=input_dict.get("embed_mask", None)
        # q_reps=self.model(features[0])
        # d_reps=self.model(features[1])
        # d_reps_neg=None
        # if len(features)>2:
        #   d_reps_neg=self.model(features[2])

        # sentence_feature={
        #   "input_ids": input_dict.get["input_ids"], #q_reps,
        #   "attention_mask": input_dict.get["attension_mask"],
        #   "embed_mask": input_dict.get["embed_mask"]
        # }
        # if d_reps_neg is not None:
        #   sentence_feature["embed_mask"]=d_reps_neg
        # for key in input_dict:
        #   input_dict[key]=input_dict[key].to(dtype=torch.float)
        input_dict["input_ids"]=input_dict["input_ids"].to(dtype=torch.long)
        outputs=model(input_dict)#(sentence_feature)#(input_ids)#, embed_mask"=embed_mask)#, attention_mask=attention_mask)#model(**input_dict)
        print("outputs:", outputs)
        print(outputs.shape)
        representation=outputs#.last_hidden_state[:, 0, :]
        representation=representation.to(dtype=torch.long)#dtype=torch.float)
        predictions=self.mlp(representation).squeeze(-1)
        loss=nn.MSELoss()(predictions, labels.float())
        if return_outputs:
            return (loss, outputs)
        return loss

    def _save(self, output_dir: Optional[str]=None, state_dict=None):
        output_dir="/content/drive/MyDrive/llm2vec-main2"#output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.model.save_pretrained(output_dir)
        torch.save(self.mlp.state_dict(), os.path.join(output_dir, "mlp.pt"))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

from transformers import HfArgumentParser, TrainingArguments
from accelerate import Accelerator
from transformers.trainer import set_seed


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            custom_args,
        ) = parser.parse_args_into_dataclasses()
    if training_args.ddp_find_unused_parameters:
        kwargs = [
            DistributedDataParallelKwargs(
                dim=0,
                broadcast_buffers=True,
                bucket_cap_mb=25,
                find_unused_parameters=True,
                check_reduction=False,
                gradient_as_bucket_view=False,
            )
        ]
    else:
        kwargs = []
    accelerator = Accelerator(kwargs_handlers=kwargs)

    set_seed(training_args.seed)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # train_dataset = load_dataset(
    #     data_args.dataset_name,
    #     split="train",
    #     file_path=data_args.dataset_file_path,
    # )
    train_examples=load_dataset_from_dataframe('/content/drive/MyDrive/llm2vec-main2/experiments/tfns.parquet')

    # train_examples = [
    #     train_dataset[i]
    #     for i in tqdm(
    #         range(len(train_dataset)),
    #         desc="Loading train examples...",
    #         disable=not accelerator.is_main_process,
    #     )
    # ] #label?



    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        #attention_dropout=custom_args.simcse_dropout,
    )

    # model organization is LLM2VecModel.model -> HF Model, we have to apply PEFT to the inner model
    model.model = initialize_peft(
        model.model,
        lora_r=custom_args.lora_r,
        lora_alpha=2 * custom_args.lora_r,
        lora_dropout=custom_args.lora_dropout,
    )

    tokenizer = model.tokenizer

    #train_loss = load_loss(custom_args.loss_class, scale=custom_args.loss_scale)

    data_collator = DefaultCollator(model)

    # trainer = SimCSETrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_examples,
    #     data_collator=data_collator,
    #     tokenizer=tokenizer,
    #     loss_function=train_loss,
    # )

    trainer = ReturnPredictionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
        mlp_hidden_dim=custom_args.lora_r * 64,
        #loss_function=train_loss,
    )

    if custom_args.stop_after_n_steps is not None:
        trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

    trainer.train()


if __name__ == "__main__":
    main()

  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/sparse.py", line 164, in forward
    return F.embedding(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/functional.py", line 2267, in embedding
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)





import logging
from dataclasses import dataclass, field
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import pandas as pd
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import seed_worker

from peft import LoraConfig, get_peft_model
import os
# 添加环境变量
# os.environ['MY_VARIABLE'] ="/home/rliuaj/llm2vec-main/"

import sys

sys.path.append("/content/drive/MyDrive/llm2vec-main2")


from llm2vec import LLM2Vec
from llm2vec.dataset.utils import load_dataset
from llm2vec.loss.utils import load_loss

from tqdm import tqdm
from datasets import Dataset

transformers.logging.set_verbosity_error()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def load_dataset_from_dataframe(file_path: str):
    df=pd.read_parquet(file_path) #'/home/rliuaj/text/gpt_api/tfns.parquet'
    # dataset=Dataset.from_pandas(df)
    # return dataset
    examples=[{"input":row["input"], "output":row["output"]} for _,row in df.iterrows()]
    return examples


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
        "GemmaConfig",
        "Qwen2Config",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The PEFT model checkpoint to add on top of base model.")},
    )
    bidirectional: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable bidirectional attention in the model. If set to False, the model will use unidirectional attention."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    pooling_mode: Optional[str] = field(
        default="mean",
        metadata={
            "help": ("The pooling mode to use in the model."),
            "choices": ["mean", "weighted_mean", "eos_token"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use. Options: E5"},
    )
    dataset_file_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file or folder."}
    )
    # TODO: implement this
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


@dataclass
class CustomArguments:
    """
    Custom arguments for the script
    """

    simcse_dropout: float = field(
        default=0.1, metadata={"help": "The SimCSE dropout rate for the model"}
    )

    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )

    lora_r: int = field(default=8, metadata={"help": "The r value for lora"})

    stop_after_n_steps: int = field(
        default=10000, metadata={"help": "Stop training after n steps"}
    )

    experiment_id: Optional[str] = field(
        default=None, metadata={"help": "The experiment id"}
    )

    loss_class: Optional[str] = field(
        default="HardNegativeNLLLoss",
        metadata={
            "help": "The loss class to use for training. Options: HardNegativeNLLLoss"
        },
    )

    loss_scale: float = field(
        default=50.0, metadata={"help": "The loss scale for the loss function"}
    )


# @dataclass
# class DefaultCollator:
#     model: LLM2Vec

#     def __init__(self, model: LLM2Vec) -> None:
#         self.model = model

#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#         batch = features
#         num_texts = len(batch[0]['input'])
#         texts = [[] for _ in range(num_texts)]
#         labels = []

#         for example in batch:
#             for idx, text in enumerate(example['input']):
#                 # TODO: Add prepare_for_tokenization here similar to supervised training and see if it impacts performance
#                 texts[idx].append(text)
#             labels.append(example['output'])
#         labels = torch.tensor(labels)

#         sentence_features = []
#         for idx in range(num_texts):
#             tokenized = self.model.tokenize(texts[idx])
#             sentence_features.append(tokenized)

#         return sentence_features, labels

@dataclass
class DefaultCollator:
    model: LLM2Vec

    def __init__(self, model: LLM2Vec) -> None:
        self.model = model

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts=[example['input'] for example in features]
        labels=torch.tensor([example['output'] for example in features], dtype=torch.float)
        
        # sentence_features=[]
        # for text in texts:
        #   tokenized=self.model.tokenize([texts])
          # sentence_features.append(tokenized)
        #tokenized=self.model.tokenize(texts)
        # sentence_features = []
        # for idx in range(len(labels)):
        #     tokenized = self.model.tokenize(texts[idx])
        #     sentence_features.append(tokenized)
        tokenized_features = self.model.tokenize(texts)
        return tokenized_features, labels
        #{"input_ids":tokenized["input_ids"],"attention_mask":tokenized["attention_mask"],
         # "labels":labels}#sentence_features, labels


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


# class SimCSETrainer(Trainer):

#     def __init__(
#         self,
#         *args,
#         loss_function=None,
#         **kwargs,
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         self.loss_function = loss_function

#     def compute_loss(
#         self,
#         model: nn.Module,
#         inputs: Dict[str, Union[torch.Tensor, Any]],
#         return_outputs: bool = False,
#     ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
#         features, labels = inputs
#         q_reps = self.model(features[0])
#         d_reps = self.model(features[1])

#         d_reps_neg = None
#         if len(features) > 2:
#             d_reps_neg = self.model(features[2])

#         loss = self.loss_function(q_reps, d_reps, d_reps_neg)

#         if return_outputs:
#             output = torch.cat(
#                 [model(row)["sentence_embedding"][:, None] for row in features], dim=1
#             )
#             return loss, output

#         return loss

#     def _save(self, output_dir: Optional[str] = None, state_dict=None):
#         # If we are executing this function, we are the process zero, so we don't check for that.
#         output_dir = output_dir if output_dir is not None else self.args.output_dir
#         os.makedirs(output_dir, exist_ok=True)
#         logger.info(f"Saving model checkpoint to {output_dir}")

#         self.model.save(output_dir)

#         # Good practice: save your training arguments together with the trained model
#         torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

class ReturnPredictionTrainer(Trainer):
    def __init__(
        self,
        *args,
        mlp_hidden_dim:int=4096,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mlp=nn.Sequential(
            nn.Linear(self.model.config.hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mlp.to(device)#(self.model.device)
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str,Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) ->Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        print("print inputs: ",inputs)
        #labels=inputs.pop("labels")
        input_dict, labels=inputs
        embed_mask=input_dict.pop("embed_mask",None)
        #input_ids=input_dict["input_ids"]
        #attention_mask=input_dict.get("attension_mask", None)
        #embed_mask=input_dict.get("embed_mask", None)
        # q_reps=self.model(features[0])
        # d_reps=self.model(features[1])
        # d_reps_neg=None
        # if len(features)>2:
        #   d_reps_neg=self.model(features[2])

        # sentence_feature={
        #   "input_ids": input_dict.get["input_ids"], #q_reps,
        #   "attention_mask": input_dict.get["attension_mask"],
        #   "embed_mask": input_dict.get["embed_mask"]
        # }
        # if d_reps_neg is not None:
        #   sentence_feature["embed_mask"]=d_reps_neg
        # for key in input_dict:
        #   input_dict[key]=input_dict[key].to(dtype=torch.float)
        input_dict["input_ids"]=input_dict["input_ids"].to(dtype=torch.long)
        outputs=model(input_dict)#(sentence_feature)#(input_ids)#, embed_mask"=embed_mask)#, attention_mask=attention_mask)#model(**input_dict)
        print("outputs:", outputs)
        print(outputs.shape)
        representation=outputs#.last_hidden_state[:, 0, :]
        representation=representation.to(dtype=torch.long)#dtype=torch.float)
        predictions=self.mlp(representation).squeeze(-1)
        loss=nn.MSELoss()(predictions, labels.float())
        if return_outputs:
            return (loss, outputs)
        return loss

    def _save(self, output_dir: Optional[str]=None, state_dict=None):
        output_dir="/content/drive/MyDrive/llm2vec-main2"#output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.model.save_pretrained(output_dir)
        torch.save(self.mlp.state_dict(), os.path.join(output_dir, "mlp.pt"))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

from transformers import HfArgumentParser, TrainingArguments
from accelerate import Accelerator
from transformers.trainer import set_seed


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            custom_args,
        ) = parser.parse_args_into_dataclasses()
    if training_args.ddp_find_unused_parameters:
        kwargs = [
            DistributedDataParallelKwargs(
                dim=0,
                broadcast_buffers=True,
                bucket_cap_mb=25,
                find_unused_parameters=True,
                check_reduction=False,
                gradient_as_bucket_view=False,
            )
        ]
    else:
        kwargs = []
    accelerator = Accelerator(kwargs_handlers=kwargs)

    set_seed(training_args.seed)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # train_dataset = load_dataset(
    #     data_args.dataset_name,
    #     split="train",
    #     file_path=data_args.dataset_file_path,
    # )
    train_examples=load_dataset_from_dataframe('/content/drive/MyDrive/llm2vec-main2/experiments/tfns.parquet')

    # train_examples = [
    #     train_dataset[i]
    #     for i in tqdm(
    #         range(len(train_dataset)),
    #         desc="Loading train examples...",
    #         disable=not accelerator.is_main_process,
    #     )
    # ] #label?



    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        #attention_dropout=custom_args.simcse_dropout,
    )

    # model organization is LLM2VecModel.model -> HF Model, we have to apply PEFT to the inner model
    model.model = initialize_peft(
        model.model,
        lora_r=custom_args.lora_r,
        lora_alpha=2 * custom_args.lora_r,
        lora_dropout=custom_args.lora_dropout,
    )

    tokenizer = model.tokenizer

    #train_loss = load_loss(custom_args.loss_class, scale=custom_args.loss_scale)

    data_collator = DefaultCollator(model)

    # trainer = SimCSETrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_examples,
    #     data_collator=data_collator,
    #     tokenizer=tokenizer,
    #     loss_function=train_loss,
    # )

    trainer = ReturnPredictionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
        mlp_hidden_dim=custom_args.lora_r * 64,
        #loss_function=train_loss,
    )

    if custom_args.stop_after_n_steps is not None:
        trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

    trainer.train()


if __name__ == "__main__":
    main()




    loss = self.compute_loss(model, inputs)
  File "/content/drive/MyDrive/llm2vec-main2/experiments/run_simcse.py", line 382, in compute_loss
    predictions=self.mlp(representation).squeeze(-1)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", line 219, in forward
    input = module(input)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/linear.py", line 117, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Float
  0%|                                                   | 0/225 [00:02<?, ?it/s]





     main()
  File "/content/drive/MyDrive/llm2vec-main2/experiments/run_simcse.py", line 509, in main
    trainer.train()
  File "/usr/local/lib/python3.10/dist-packages/transformers/trainer.py", line 1932, in train
    return inner_training_loop(
  File "/usr/local/lib/python3.10/dist-packages/transformers/trainer.py", line 2345, in _inner_training_loop
    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
  File "/usr/local/lib/python3.10/dist-packages/transformers/trainer.py", line 2796, in _maybe_log_save_evaluate
    self._save_checkpoint(model, trial, metrics=metrics)
  File "/usr/local/lib/python3.10/dist-packages/transformers/trainer.py", line 2875, in _save_checkpoint
    self.save_model(output_dir, _internal_call=True)
  File "/usr/local/lib/python3.10/dist-packages/transformers/trainer.py", line 3429, in save_model
    self._save(output_dir)
  File "/content/drive/MyDrive/llm2vec-main2/experiments/run_simcse.py", line 392, in _save
    self.model.save_pretrained(output_dir)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1729, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'LLM2Vec' object has no attribute 'save_pretrained'. Did you mean: 'from_pretrained'?
