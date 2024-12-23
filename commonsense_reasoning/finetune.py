# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import dataclasses
import os
import sys
from abc import ABC
from dataclasses import field
from typing import List, Union

import fire
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import (  # noqa: E402
    LoraConfig,
    DoraConfig,
    BiDoraConfig,
    BottleneckConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer  # noqa: F402


@dataclasses.dataclass
class PEFTTrainingArguments(transformers.TrainingArguments):
    outer_learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    outer_weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    target_modules: List = field(default=None)


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "lora",
        load_8bit: bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        outer_learning_rate: float = 3e-4,
        outer_weight_decay: float = 0.0,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        # Dora hyperparams
        dora_simple: bool = True,
        Wdecompose_target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        bilevel: bool = False
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"Wdecompose_target_modules: {Wdecompose_target_modules}\n"
        f"dora_simple: {dora_simple}"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            # torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            # torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )

    if model.config.model_type == "llama":
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        # need to handle llama 3 separately
        if "Llama-3" in base_model:
            print("load llama-3 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    print(model)
    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "dora":
        print("DoRA init")
        config = DoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            dora_simple=dora_simple,
            Wdecompose_target_modules=Wdecompose_target_modules
        )
    elif adapter_name == "bidora":
        print("BiDoRA init")
        config = BiDoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            dora_simple=dora_simple,
            Wdecompose_target_modules=Wdecompose_target_modules
        )
    elif adapter_name == "bottleneck":
        config = BottleneckConfig(
            bottleneck_size=bottleneck_size,
            non_linearity=non_linearity,
            adapter_dropout=adapter_dropout,
            use_parallel_adapter=use_parallel_adapter,
            use_adapterp=use_adapterp,
            target_modules=target_modules,
            scaling=scaling,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "prefix-tuning":
        config = PrefixTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
        )
    else:
        raise NotImplementedError(f'Unknown adapter_name {adapter_name}')
    # model.add_adapter(config)
    model = get_peft_model(model, config)
    if adapter_name == "prefix-tuning":
        model.to('cuda')

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    print(f'Training dataset size: {len(train_data)}')
    print(f'Validation dataset size: {len(val_data)}')
    print('PEFT model', model.__class__.__name__)
    print('PEFT base model', model.base_model.__class__.__name__)
    if bilevel:
        train_split = train_data.train_test_split(test_size=0.2, shuffle=True)
        inner_train_data, outer_train_data = train_split['train'], train_split['test']
        print(f'Inner training dataset size: {len(inner_train_data)}')
        print(f'Outer training dataset size: {len(outer_train_data)}')
        trainer = BiDoRATrainer(
            model=model,
            train_dataset=inner_train_data,
            outer_train_dataset=outer_train_data,
            eval_dataset=val_data,
            args=PEFTTrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=num_epochs,
                max_steps=20000,
                eval_steps=200,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                outer_learning_rate=outer_learning_rate,
                outer_weight_decay=outer_weight_decay,
                fp16=True,
                logging_steps=10,
                save_steps=save_step,
                output_dir=output_dir,
                group_by_length=group_by_length,
                run_name=wandb_run_name if use_wandb else None,
                target_modules=target_modules
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            )
        )
        model.config.use_cache = False

        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        for name, param in model.named_parameters():
            print(f"Llama (DoRA) Parameter: {name} | Type: {param.type()}")
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=PEFTTrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                fp16=True,
                logging_steps=10,
                optim="adamw_torch",
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=eval_step if val_set_size > 0 else None,
                save_steps=save_step,
                output_dir=output_dir,
                save_total_limit=3,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            )
        )
        model.config.use_cache = False

        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        # trainer.push_to_hub()
        print(f'Saving to {output_dir}')
        # model.save_pretrained(output_dir)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        model.save_pretrained(output_dir)

        print(
            "\n If there's a warning about missing keys above, please disregard :)"
        )


from transformers.trainer import *
from betty.engine import Engine
from betty.problems import ImplicitProblem
import wandb
from betty.utils import convert_tensor


class BiDoRAProblem(ImplicitProblem, ABC):
    def get_batch_single_loader(self, idx):
        """
        Load training batch from one of the user-provided data loader(s)

        :return: New training batch
        :rtype: Any
        """
        data_iterator = self.train_data_iterator[idx]
        try:
            batch = next(data_iterator)
        except StopIteration:
            if idx == 0:
                self.epoch_callback_exec()
            self.epoch_counter[idx] += 1
            train_data_loader = self.train_data_loader[idx]
            if self._strategy in ["distributed", "zero", "fsdp"]:
                train_data_loader.set_epoch(self.epoch_counter[idx])
            self.train_data_iterator[idx] = iter(train_data_loader)
            batch = next(self.train_data_iterator[idx])
        # print('In get_batch_single_loader', batch)
        # print(type(batch))
        # if not isinstance(batch, dict):
        #     print('batch is not dict')
        #     batch = tuple(convert_tensor(value, self.device) for value in batch)
        # else:
        for key, value in batch.items():
            batch[key] = convert_tensor(value, self.device)
        # print('return from get_batch_single_loader', batch)
        return batch

    def optimizer_step(self, *args, **kwargs):
        if self.is_implemented("custom_optimizer_step"):
            if self.gradient_clipping > 0.0:
                self.clip_grad()
            self.custom_optimizer_step(*args, **kwargs)
        else:
            if self.scaler is not None:
                # self.scaler.unscale_(self.optimizer)
                if self.gradient_clipping > 0.0:
                    self.clip_grad()
                self.scaler.step(self.optimizer)
                if self.config.type in ["sama"]:
                    for param in self.trainable_parameters():
                        state = self.get_opt_state_for_param(param)
                        if param.grad is not None and len(state) != 0:
                            state["last_grad"] = param.grad.detach().clone()
                self.scaler.update()
            else:
                if self.gradient_clipping > 0.0:
                    self.clip_grad()
                self.optimizer.step()
                if self.config.type in ["sama"]:
                    for param in self.trainable_parameters():
                        state = self.get_opt_state_for_param(param)
                        if param.grad is not None and len(state) != 0:
                            state["last_grad"] = param.grad.detach().clone()


class Inner(BiDoRAProblem):
    def training_step(self, batch):
        # print(batch)
        print('Inner Problem')
        batch = {key: value.to(self.device) for key, value in batch.items()}
        loss = self.module(**batch, return_dict=True, alphas=self.outer()).loss
        wandb.log({
            "Inner/batch loss": loss.cpu().item(),
            "Inner/loss": loss.cpu().item(),
            "Inner/lr": self.optimizer.param_groups[0]['lr']
        }, step=self._global_step)
        return loss


class Outer(BiDoRAProblem):
    def training_step(self, batch):
        print('Outer Problem')
        batch = {key: value.to(self.device) for key, value in batch.items()}
        loss = self.inner.module(**batch, return_dict=True, alphas=self.forward()).loss
        wandb.log({
            "Inner/batch loss": loss.cpu().item(),
            "Inner/loss": loss.cpu().item(),
            "Inner/lr": self.optimizer.param_groups[0]['lr']
        }, step=self._global_step)
        return loss


class BilevelEngine(Engine):

    @torch.no_grad()
    def validation(self):
        print('validation')


from dataclasses import asdict

from peft.tuners.bidora import LoraLayer


def is_bidora_layer(module):
    # print('Checking bidora layer...')
    # print(f'Module name {module.__class__.__name__}')
    # return 'bidora' in module.__class__.__name__
    return isinstance(module, LoraLayer)


class DoRAMagnitude(nn.Module):
    def __init__(self, magnitude):
        super(DoRAMagnitude, self).__init__()
        self.magnitude = nn.Parameter(magnitude.clone(), requires_grad=True)


class BiDoRAArchitecture(torch.nn.Module):

    def __init__(self, model, target_modules):
        super(BiDoRAArchitecture, self).__init__()
        print(f'Initializing bidora architecture with target modules {target_modules}')
        magnitude_lists = {module: [] for module in target_modules}

        for module_name, module in model.named_modules():
            if is_bidora_layer(module):
                with torch.no_grad():
                    magnitude = torch.linalg.norm(module.weight, dim=1).unsqueeze(1)

                print(f'Initialize from module {module_name}')
                for target_module in target_modules:
                    if target_module in module_name:
                        magnitude_lists[target_module].append(magnitude)
                        print(f'{module_name} added to {target_module} list!')

        lengths = [len(lst) for lst in magnitude_lists.values()]
        assert len(set(lengths)) == 1, "All target_module lists must have the same length"
        num_layers = lengths[0]

        self.magnitudes = nn.ModuleList()
        for i in range(num_layers):
            layer_dict = nn.ModuleDict()
            for module in target_modules:
                layer_dict[module] = DoRAMagnitude(magnitude_lists[module][i])
            self.magnitudes.append(layer_dict)

        print(f'BiDoRAArchitecture initialized with {num_layers} layers.')

    def forward(self):
        return self.magnitudes


class BiDoRATrainer(transformers.Trainer):

    def __init__(self, outer_train_dataset=None, **kwargs):
        super().__init__(**kwargs)
        self.args: PEFTTrainingArguments = kwargs['args']
        self.outer_train_dataset = outer_train_dataset
        self.alphas = BiDoRAArchitecture(self.model, self.args.target_modules)
        print('BiDoRA architecture alphas')
        print(self.alphas)
        print("alphas' parameter list size")
        print(len(list(self.alphas.parameters())))

        for name, param in self.model.named_parameters():
            print(f"Inner Parameter: {name} | Type: {param.type()}")
        for name, param in self.alphas.named_parameters():
            print(f"Outer Parameter: {name} | Type: {param.type()}")
        wandb.init(project='dora', name='bidora commonsense reasoning', config=asdict(self.args))

    def get_train_dataloader(self, dataset=None) -> DataLoader:
        if dataset is None:
            raise ValueError("Trainer: training requires a dataset.")

        data_collator = self.data_collator
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            # dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["sampler"] = RandomSampler(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = None

        # return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))
        return DataLoader(dataset, **dataloader_params)

    def count_optimized_parameters(self, param_groups):
        total_params = 0
        for param_group in param_groups:
            for param in param_group['params']:
                total_params += param.numel()
        return total_params

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """

        inner_params_list = []
        outer_params_list = []
        for name, param in self.model.named_parameters():

            if 'weight_m_wdecomp' in name:
                outer_params_list.append(param)
                print(f'{name} is trainable at upper level')
            elif 'lora_' in name:
                inner_params_list.append(param)
                print(f'{name} is trainable at lower level')
            else:
                print(f'{name} is not trainable')

        print("Using alphas' parameters")
        outer_params_list = list(self.alphas.parameters())

        inner_parameter_groups = [{
            "params": inner_params_list, "weight_decay": self.args.weight_decay,
        }]
        outer_parameter_groups = [{
            "params": outer_params_list, "weight_decay": self.args.outer_weight_decay,
        }]
        from torch.optim import AdamW, SGD, ASGD, Rprop

        optimizer_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "fused": True
        }
        # inner_optimizer = AdamW(inner_parameter_groups, **{**optimizer_kwargs, "lr": self.args.learning_rate})
        # outer_optimizer = AdamW(outer_parameter_groups, **{**optimizer_kwargs, "lr": self.args.outer_learning_rate})
        inner_optimizer = SGD(inner_parameter_groups, **{"lr": self.args.learning_rate})
        outer_optimizer = SGD(outer_parameter_groups, **{"lr": self.args.outer_learning_rate})
        print('#Inner params', self.count_optimized_parameters(inner_parameter_groups))
        print('#Outer params', self.count_optimized_parameters(outer_parameter_groups))
        return inner_optimizer, outer_optimizer

    def create_scheduler(self, num_training_steps: int, inner_optimizer: torch.optim.Optimizer = None,
                         outer_optimizer: torch.optim.Optimizer = None):

        inner_lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=inner_optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
        )

        outer_lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=inner_optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
        )

        return inner_lr_scheduler, outer_lr_scheduler

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            ignore_keys_for_eval: Optional[List[str]] = None,
            **kwargs,
    ):

        for param in self.model.parameters():
            param.requires_grad = True
        from betty.configs import EngineConfig, Config
        inner_optimizer, outer_optimizer = self.create_optimizer()
        inner_scheduler, outer_scheduler = self.create_scheduler(
            self.args.max_steps, inner_optimizer, outer_optimizer)
        precision = 'fp16'
        # precision = 'fp32'
        inner_config = Config(type="darts", unroll_steps=1, gradient_accumulation=1, precision=precision)
        outer_config = Config(type="darts", retain_graph=True, gradient_accumulation=1, precision=precision)
        engine_config = EngineConfig(
            train_iters=self.args.max_steps, valid_step=self.args.eval_steps)
        inner_dataloader = self.get_train_dataloader(dataset=self.train_dataset)
        outer_dataloader = self.get_train_dataloader(dataset=self.outer_train_dataset)
        sample_batch = next(iter(inner_dataloader))
        print('Sample batch from inner loader')
        print(sample_batch.keys())
        print(sample_batch)
        inner = Inner(name="inner", module=self.model, optimizer=inner_optimizer, scheduler=inner_scheduler,
                      config=inner_config, train_data_loader=inner_dataloader)
        outer = Outer(name="outer", module=self.alphas, optimizer=outer_optimizer, scheduler=outer_scheduler,
                      config=outer_config, train_data_loader=outer_dataloader)
        problems = [outer, inner]
        l2u = {inner: [outer]}
        u2l = {outer: [inner]}
        dependencies = {"l2u": l2u, "u2l": u2l}
        engine = BilevelEngine(config=engine_config, problems=problems, dependencies=dependencies)
        engine.run()
        self.model.save_pretrained(self.args.output_dir, magnitudes=self.alphas)


def compute_magnitude_regularization(alphas):
    regu_loss, num_param = 0., 0
    for alpha in alphas:
        for m in alpha:
            regu_loss += m.abs().sum()
            num_param += 1

    return regu_loss / num_param


def compute_direction_regularization(model):
    regu_loss, num_param = 0., 0
    for module_name, module in model.named_modules():
        if is_bidora_layer(module):
            D = module.weight()
            D_ = D.T @ D
            I = torch.eye(len(D_), device=D_.device)
            regu_loss += torch.norm(D_ - I, p="fro")
            # regu_loss += torch.linalg.matrix_norm(D_ - I, ord=2)
            num_param += 1

    return regu_loss / num_param


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}"""  # noqa: E501


if __name__ == "__main__":
    fire.Fire(train)
