import time

import os
import numpy as np
import json

import torch


from transformers import AutoTokenizer
import jsonlines

class DebugEvalDataset:
    def __init__(self, root, sample_num=1, pick_num = 3000):
        """
        root: the path to the DebugEval dataset
        sample_num: the number of samples for each prompt
        issft: whether to use the SFT setting
        """
        self.root = root
        
        self.data = []

        cnt = 0
        with jsonlines.open(os.path.join(self.root)) as reader:
            for line in reader:
                self.data.append(line)
                cnt += 1
                if cnt >= pick_num:
                    break

    def __len__(self):
        """
        return the numbers of samples in the dataset
        """
        return len(self.data)
    
    def __getitem__(self, index):
        """
        return sample at index
        """
        sample  = self.data[index]
        return sample

def calaulate_debug_accuracy(file_path):
    
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
        preds = [item["prediction_code"] for item in data]
        labels = [item["label_code"] for item in data]

    
    """
    Calculate the accuracy of the model on the DebugEval dataset.
    """
    assert len(preds) == len(labels), "The number of predictions and labels should be the same"
    correct = 0
    for i in range(len(preds)):
        if preds[i] in labels[i]:
            correct += 1
        elif labels[i] in preds[i]:
            correct += 1
    return correct / len(preds)

class DebugEval:
    """
    DebugEval Evaluation Class
    """

    def __init__(self, data_root,
                 max_seq_len=2048,
                 max_gen_len=200, 
                 batch_size=512,
                 log_dir=None, 
                 issft=True,
                 temperature=0, 
                 top_p=0.95,
                 model_name="", 
                 inference_increment=True,
                 tokenizer_cfg=None, 
                 n_sample=40, 
                 k_sample=1,
                 pick_num = 3000):
        #assign valuas
        self.data_root = data_root
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.batch_size = batch_size
        self.k = k_sample
        self.n_sample = n_sample
        self.pick_num = pick_num
        self.log_dir = log_dir
        self.sft = issft
        
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = tokenizer_cfg["model_path"].replace("/", "_")
        self.inference_increment = inference_increment
        os.makedirs(self.log_dir, exist_ok=True)
        tokenizer_cls = tokenizer_cfg.pop('cls')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg.pop("model_path"), trust_remote_code=True, max_seq_len=self.max_seq_len, max_gen_len=self.max_gen_len)       
        except Exception as e:
            print(e)
            assert False
    
    @torch.no_grad()
    def eval_model(self, gpt, accelerator):
        """
        Evaluate the model on DebugEval.
        """
        assert self.log_dir is not None, "log_dir should not be None when evaluating debugeval"
        dataset = DebugEvalDataset(self.data_root, sample_num=self.n_sample, pick_num=self.pick_num)
        nprompt = len(dataset) // self.n_sample
        dp_rank = accelerator.process_index 
        dp_size = accelerator.num_processes 
        # if self.k > 1:
        #     assert self.n_sample >= 100, "HumanEval PASS@100 needs n_sample >= 100"
        gpt.eval()

        # each process will process a subset of the dataset
        prompt_indices_split = np.array_split(range(nprompt), dp_size)
        prompt_indices = prompt_indices_split[dp_rank]
        indices = [x * self.n_sample + j for x in prompt_indices for j in range(self.n_sample)]
        all_num = len(indices) 
        processed_num = 0
        log_file = os.path.join(self.log_dir,
                                    f'{self.model_name}_rank{dp_rank}_bs{self.batch_size}_shot_log.json')
        tmpfile = open(log_file, "w")
        start_time = time.time()

        prompt_template= '#Generate Debugged Code given the following high-level-synthesis bug type and buggy code\n'
        prompt_template += "The Bug types are: \n"
        prompt_template += """
        'OOB': Out-of-bounds array access
        'INIT': Read of uninitialized variable
        'SHFT': Bit shift by an out-of-bounds amount
        'INF': An infinite loop arising from an incorrect loop termination
        'USE': Unintended sign extension
        'MLU': Errors in manual loop unrolling
        'ZERO': Variable initialized to zero instead of nonzero initializer 
        'BUF': Copying from the wrong half of a split buffer
        'APT': Array partition type error for the pragma '#pragma HLS array_partition'
        'FND': Factor not divisible for the pragma '#pragma HLS array_partition'
        'DID': Dim incorrect dimension for the pragma '#pragma HLS array_partition'
        'DFP': Dataflow position error for the pragma '#pragma HLS dataflow'
        'IDAP': Incorrect data access pattern for the pragma '#pragma HLS interface'
        'RAMB': m_axi interface is accessed randomly in the code, resulting in non-burst AXI read/write for the pragma '#pragma HLS interface'
        'SMA': Scalar values mapped to array interfaces like bram/ap_memory/m_axi/ap_fifo/axis for the pragma '#pragma HLS interface'
        'AMS': Array value mapped to scalar interfaces like ap_none/ap_vld/s_axilite for the pragma '#pragma HLS interface'
        'MLP': Multi-level pipelining for the pragma '#pragma HLS pipeline'
        'ILNU': Inner loops not fully-Unrolled for the pragma '#pragma HLS unroll'
        """

        prompt_template += "The code is: \n"

        # split the dataset into batches and construct a list of inputs
        for idx in range(0, len(indices), self.batch_size):
            prompt_list = []
            prompt_lens = []
            orriginal_prompt_list = []
            tokenized_prompt_lens = []
            label_list = []
            # taskid = []
            # get the prompts from the dataset
            for j in indices[idx:idx + self.batch_size]:
                data = dataset[j]
                # print("check type of data",type(data))
                # fprompt = data["prompt"].strip()
                fprompt = prompt_template + data["modified_code"].strip()
                prompt_list.append(fprompt)
                tmp = self.tokenizer.encode(fprompt)
                orriginal_prompt_list.append(data["modified_code"])
                label_list.append(data["source_code"])
                prompt_lens.append(len(fprompt))
                tokenized_prompt_lens.append(tmp)
                # taskid.append(data["task_id"])

                # Clear CUDA cache to avoid out-of-memory error
                torch.cuda.empty_cache()
            input_ids = torch.tensor(tokenized_prompt_lens).to(accelerator.device)
            # generate the code
            if self.temperature != 0:

                decoded = gpt.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_gen_len,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                try:
                    decoded = gpt.generate(
                        input_ids=input_ids,
                        max_new_tokens=self.max_gen_len,
                        do_sample=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                except Exception as e:
                    print(e)
                    print("Error in generating code")
                    print(orriginal_prompt_list)
            # save the results to a file
           
            for local_idx, text in enumerate(decoded):
                prediction = decoded[local_idx]
                prediction = self.tokenizer.decode(prediction, skip_special_tokens=True)
                suffixprediction = prediction[prompt_lens[local_idx]:]
                # suffixprediction = cleanup_code(suffixprediction, self.language, "humaneval", self.sft, dataset.stopwords)
                # sft mode does not need original prompt
                if not self.sft:
                    suffixprediction = orriginal_prompt_list[local_idx] + "\n" + suffixprediction
                # res = {"task_id": taskid[local_idx], "generation": suffixprediction, "prompt": orriginal_prompt_list[local_idx], "wholecode":prediction}
                res = {"generation": suffixprediction, "buggy_code": orriginal_prompt_list[local_idx], "prediction_code":prediction, "label":label_list[local_idx]}
                tmpfile.write(json.dumps(res) + "\n")
                tmpfile.flush()
                processed_num += 1
            self.log_score(dp_rank, processed_num, all_num, start_time, self.batch_size)
        tmpfile.close()        
        accelerator.wait_for_everyone()
        # calculate the final score of pass@k
        self._calculate_final_score(accelerator)
        accelerator.wait_for_everyone()
        return

    def log_score(self, dp_rank, processed_num, all_num, start_time, bs):
        """
        Log the score.
        """
        mem = torch.cuda.max_memory_allocated() / (1 << 30)
        avg_time = (time.time() - start_time) / processed_num * bs
        print(
            f'DP RANK:{dp_rank} process_num/all_num:{int(processed_num)}/{all_num} '
            f'avg_time_per_batch:{avg_time:.2f} s '
            f'still_need:{((all_num - processed_num) // bs + 1) * avg_time / 60:.2f} m',
            f'mem:{mem:.3f} GiB bs:{bs}',
            flush=True
        )
        if processed_num == all_num:
            print(f'EVAL DONE! Process time {(time.time() - start_time) / 60:.2f} m', flush=True)
    
    def _calculate_final_score(self, accelerator):
        """
        Calculate the final score.
        """
        # log_file = os.path.join(self.log_dir,
        #                             f'{self.model_name}_rank{dp_rank}_bs{self.batch_size}_shot_log_{self.language}.json')
        if accelerator.is_local_main_process:
            logfilepath = os.path.join(self.log_dir, f'final_{self.model_name}.jsonl')
            logfile = open(logfilepath, "w")
            for i in range(accelerator.num_processes):
                tmplogfile = os.path.join(self.log_dir, f'{self.model_name}_rank{i}_bs{self.batch_size}_shot_log.json')
                logfile.write(open(tmplogfile).read().strip() + "\n")
                # os.remove(tmplogfile)
            logfile.close()
            # timeout = 10
            
            # res = evaluate_functional_correctness(input_file=logfilepath, problem_file=os.path.join(self.data_root, f"humaneval-{self.language}.jsonl"), tmp_dir=self.log_dir, timeout=timeout, language=runlang)
            
            res = calaulate_debug_accuracy(tmplogfile)
            print("final score is", res)
            # print("score is", res['pass@%d' % self.k])
            # os.remove(logfilepath)
        return
