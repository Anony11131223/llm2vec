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
