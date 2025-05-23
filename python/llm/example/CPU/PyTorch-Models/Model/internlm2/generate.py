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
import time
import argparse
import numpy as np

from bigdl.llm import optimize_model
from transformers import AutoTokenizer

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/internlm/internlm-chat-7b-8k/blob/main/modeling_internlm.py#L768
INTERNLM_PROMPT_FORMAT = "<|User|>:{prompt}\n<|Bot|>:"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for InternLM model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="internlm/internlm2-chat-7b",
                        help='The huggingface repo id for the InternLM model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="AI是什么？",
                        help='Prompt to infer') 
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path


    from bigdl.llm import optimize_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True)
    model = optimize_model(model)

    

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    
    # Generate predicted tokens
    with torch.inference_mode():
        prompt = INTERNLM_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        st = time.time()
    
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        end = time.time()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        output_str = output_str.split("<eoa>")[0]
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)
