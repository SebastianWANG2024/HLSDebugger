from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
# input_text = "#write a quick sort algorithm"
# inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_length=128)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json
import torch.distributed as dist
import subprocess
import sys
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from pathlib import Path
from argparse import ArgumentParser

from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines
from datasets import load_dataset
import sys
print(sys.path)

from DebugEval import DebugEval

if __name__ == '__main__':
    
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=kwargs_handlers)   

    print("Device:", accelerator.device)
    print("Device count:", accelerator.num_processes)

    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default="")
    
    parser.add_argument("--dataroot", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--max-seq-len",type=int, default=2048)
    parser.add_argument("--max-gen-len",type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--instruction_template", type=str, default='zero_shot')
    # parser.add_argument("--pick-num", default=100)
    args = parser.parse_args()

    logdir = args.logdir
    

    if logdir == "":
        logdir = "tmp/"
    

    dataroot = args.dataroot
    
    evaluator = DebugEval(model_name=args.model_path,data_root=dataroot, log_dir=logdir, n_sample=8, batch_size=1, max_seq_len=args.max_seq_len, max_gen_len=args.max_gen_len, temperature=args.temperature)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=accelerator.device, trust_remote_code=True, torch_dtype=torch.bfloat16)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # evaluator.eval_model(model, accelerator)
    evaluator.eval_test_dataset(model, accelerator, eval_template=args.instruction_template)
    # evaluator.eval_train_dataset(model, accelerator)
    # evaluator.eval_validation_dataset(model, accelerator)

