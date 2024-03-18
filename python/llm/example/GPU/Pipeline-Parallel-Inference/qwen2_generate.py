
#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import intel_extension_for_pytorch as ipex
import time
import argparse

from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from benchmark_util import BenchmarkWrapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="/mnt/disk1/models/Qwen1.5-14B-Chat",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--n-prompt', type=int, default=1024,
                        help='Tokens of Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=128,
                        help='Max tokens to predict')
    parser.add_argument('--low-bit', type=str, default='sym_int4',
                        help='Low bit optimization')
    parser.add_argument('--batch-size', type=int, default=1)

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    in_len = args.n_prompt
    out_len = args.n_predict
    low_bit = args.low_bit
    batch_size = args.batch_size


    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_low_bit=low_bit,
                                                 optimize_model=True,
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True,
                                                 use_cache=True)
    print(model)

    first_half = ['model.embed_tokens', 'model.layers.0', 'model.layers.1', 'model.layers.2',
                  'model.layers.3', 'model.layers.4', 'model.layers.5', 'model.layers.6',
                  'model.layers.7', 'model.layers.8', 'model.layers.9', 'model.layers.10',
                  'model.layers.11', 'model.layers.12', 'model.layers.13', 'model.layers.14',
                  'model.layers.15', 'model.layers.16', 'model.layers.17', 'model.layers.18',
                  'model.layers.19']
    second_half = ['model.layers.20', 'model.layers.21', 'model.layers.22', 'model.layers.23',
                   'model.layers.24', 'model.layers.25', 'model.layers.26', 'model.layers.27',
                   'model.layers.28', 'model.layers.29', 'model.layers.30', 'model.layers.31',
                   'model.layers.32', 'model.layers.33', 'model.layers.34', 'model.layers.35',
                   'model.layers.36', 'model.layers.37', 'model.layers.38', 'model.layers.39',
                   'model.norm', 'lm_head']

    device_map=({key: 'xpu:0' for key in first_half})
    device_map.update({key: 'xpu:1' for key in second_half})
    from accelerate import dispatch_model
    model = dispatch_model(
        model,
        device_map=device_map,
        offload_dir=None,
        skip_keys=["past_key_value", "past_key_values"],
    )

    model = BenchmarkWrapper(model, do_print=True)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Generate predicted tokens
    with torch.inference_mode():
        input_str = open(f"2048.txt", 'r').read()
        input_ids = tokenizer.encode(input_str, return_tensors="pt")
        input_ids = input_ids[:, :in_len]
        true_str = tokenizer.batch_decode(input_ids)[0]
        input_list = [true_str] * batch_size
        input_ids = tokenizer(input_list, return_tensors="pt").input_ids.to('xpu:0')
        actual_in_len = input_ids.shape[1]

        # ipex model needs a warmup, then inference time can be accurate
        output_ids = model.generate(input_ids,
                                    do_sample=False,
                                    num_beams=1,
                                    max_new_tokens=out_len)
        output_ids = model.generate(input_ids,
                                    do_sample=False,
                                    num_beams=1,
                                    max_new_tokens=out_len)

        torch.xpu.synchronize()
        # start inference
        for i in range(3):
            st = time.time()
            # if your selected model is capable of utilizing previous key/value attentions
            # to enhance decoding speed, but has `"use_cache": false` in its model config,
            # it is important to set `use_cache=True` explicitly in the `generate` function
            # to obtain optimal performance with BigDL-LLM INT4 optimizations
            output_ids = model.generate(input_ids,
                                        do_sample=False,
                                        num_beams=1,
                                        max_new_tokens=out_len)
            torch.xpu.synchronize()
            end = time.time()
        output_ids = output_ids.cpu()

        output = tokenizer.batch_decode(output_ids)
        torch.xpu.empty_cache()
        actual_out_len = output_ids.shape[1] - actual_in_len
        print(f'Test case: {actual_in_len}, {actual_out_len}')
        print('====Prompt====')
        print(true_str)
        print('====Output====')
        print(output[0])
