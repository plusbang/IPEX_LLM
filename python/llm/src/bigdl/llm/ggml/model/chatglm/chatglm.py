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
# ===========================================================================
#
# This file is adapted from
# https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py
#
# MIT License
#
# Copyright (c) 2023 Andrei Betlen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This would makes sure Python is aware there is more than one sub-package within bigdl,
# physically located elsewhere.
# Otherwise there would be module not found error in non-pip's setting as Python would
# only search the first bigdl package and end up finding only one sub-package.


from .chatglm_cpp import chatglm_load, chatglm_tokenize, chatglm_detokenize, \
    chatglm_forward, chatglm_eos_token
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.ggml.model.generation import GenerationMixin
from typing import List, Optional, Generator, Sequence, Union
import time
import uuid
import warnings


class ChatGLM(GenerationMixin):
    """High-level Python wrapper for a chatglm.cpp model."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_parts: int = -1,
        n_gpu_layers: int = 0,
        seed: int = -1,
        f16_kv: bool = True,
        logits_all: bool = False,
        vocab_only: bool = False,
        use_mmap: bool = False,
        use_mlock: bool = False,
        embedding: bool = False,
        n_threads: Optional[int] = -1,
        n_batch: int = 512,
        last_n_tokens_size: int = 64,
        lora_base: Optional[str] = None,
        lora_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """Load a chatglm.cpp model from `model_path`.

        Args:
            model_path: Path to the model.
            n_ctx: Maximum context size.
            n_parts: Number of parts to split the model into. If -1, the number of parts
            is automatically determined.
            seed: Random seed. For default value -1, current timestamp is used as seed.
            f16_kv: Use half-precision for key/value cache.
            logits_all: Return logits for all tokens, not just the last token.
            vocab_only: Only load the vocabulary no weights.
            use_mmap: Use mmap if possible.
            use_mlock: Force the system to keep the model in RAM.
            embedding: Embedding mode only.
            n_threads: Number of threads to use. Default to be -1, means auto.
            n_batch: Maximum number of prompt tokens to batch together when calling chatglm_eval.
            last_n_tokens_size: Maximum number of tokens to keep in the last_n_tokens deque.
            lora_base: Optional path to base model, useful if using a quantized base model and
            you want to apply LoRA to an f16 model.
            lora_path: Path to a LoRA file to apply to the model.
            verbose: Print verbose output to stderr.

        Raises:
            ValueError: If the model path does not exist.

        Returns:
            A ChatGLM instance.
        """

        self.model_path = model_path
        self.ctx = chatglm_load(model_path, use_mmap=use_mmap, n_ctx=n_ctx, n_threads=n_threads)
        self.n_ctx = n_ctx
        self.n_parts = n_parts
        self.n_gpu_layers = n_gpu_layers
        self.f16_kv = f16_kv
        self.seed = seed
        self.logits_all = logits_all
        self.vocab_only = vocab_only
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.embedding = embedding
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.last_n_tokens_size = last_n_tokens_size
        self.lora_base = lora_base
        self.lora_path = lora_path
        self.verbose = verbose
        # TODO: Some parameters are temporarily not supported
        unsupported_arg = {'n_parts': -1, 'n_gpu_layers': 0, 'f16_kv': True, 'logits_all': False,
                           'vocab_only': False, 'use_mlock': False, 'embedding': False,
                           'n_batch': 512, 'last_n_tokens_size': 64, 'lora_base': None,
                           'lora_path': None, 'verbose': True}
        for arg in unsupported_arg.keys():
            if getattr(self, arg) != unsupported_arg[arg]:
                warnings.warn(f"The parameter {arg} is temporarily unsupported, "
                              "please use the default value.")

    def __call__(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.95,
        top_p: float = 0.7,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]]=[],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 0,
        stream: bool = False,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
    ):
        # TODO: Some parameters are temporarily not supported
        # Unsupported parameters are checked in `_supported_call`
        return self._supported_call(prompt, max_tokens, stream, temperature, top_p, top_k,
                                    stop, model, suffix, logprobs, echo, frequency_penalty,
                                    presence_penalty, repeat_penalty, tfs_z, mirostat_mode,
                                    mirostat_tau, mirostat_eta)

    def _supported_call(self, prompt: str, max_tokens: int, stream: bool,
                        temperature: float, top_p: float, top_k: int,
                        stop: Optional[List[str]] = [], model: Optional[str] = None, *args):
        # Check unsupporeted parameters
        unsupported_arg = ['suffix', 'logprobs', 'echo',
                           'frequency_penalty', 'presence_penalty', 'repeat_penalty',
                           'tfs_z', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'model']
        defult_value = {'suffix': None, 'logprobs': None, 'echo': False,
                        'frequency_penalty': 0.0, 'presence_penalty': 0.0,
                        'repeat_penalty': 1.1, 'tfs_z': 1.0, 'mirostat_mode': 0,
                        'mirostat_tau': 5.0, 'mirostat_eta': 0.1}
        for index in range(len(args)):
            if args[index] != defult_value[unsupported_arg[index]]:
                warnings.warn(f"The parameter {unsupported_arg[index]} is temporarily "
                              "unsupported, please use the default value.")

        if stream:
            return self.stream(prompt, max_tokens, temperature, top_p, top_k, stop, model)
        else:
            return self._eval(prompt, max_tokens, temperature, top_p, top_k, stop, model)

    def _eval(self, prompt: str, max_tokens: int, temperature: float, top_p: float, top_k: int,
              stop: Optional[List[str]] = [], model: Optional[str] = None):

        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        if model is None:
            model_name = self.model_path
        else:
            model_name = model

        input_tokens = self._tokenize(prompt)
        prompt_len = len(input_tokens)
        if max_tokens < 1:
            return {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "text": prompt,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "length",
                    }
                ],
                "usage":
                {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": 0,
                    "total_tokens": prompt_len,
                }
            }

        for i in range(max_tokens):
            token = self.forward(input_ids=input_tokens,
                                 top_k=top_k,
                                 top_p=top_p,
                                 temperature=temperature)
            input_tokens.append(token)
            if token == self.eos_token():
                break

        text = self.detokenize(input_tokens)
        split_text = text[len(prompt):]
        split_text.rstrip('�')  # remove partial emoji
        if stop != []:
            for stop_word in stop:
                split_text = split_text.split(stop_word)[0]
        if split_text != text:
            finish_reason = "stop"
        else:
            finish_reason = None
        completion_len = len(input_tokens) - prompt_len
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": prompt + split_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_len,
                "total_tokens": prompt_len + completion_len,
            }
        }

    def stream(self, prompt: str, max_tokens: int, temperature: float, top_p: float, top_k: int,
               stop: Optional[List[str]] = [], model: Optional[str] = None):
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        if model is None:
            model_name = self.model_path
        else:
            model_name = model
        input_tokens = self._tokenize(prompt)
        prompt_len = len(input_tokens)
        if max_tokens < 1:
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "text": prompt,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_len
                }
            }
        else:
            history_text = prompt
            for i in range(max_tokens):
                token = self.forward(input_ids=input_tokens,
                                     top_k=top_k,
                                     top_p=top_p,
                                     temperature=temperature)
                input_tokens.append(token)
                if token == self.eos_token():
                    print('\n')
                    break
                text = self.detokenize(input_tokens)
                if text.endswith('�'):
                    # generated new token is part of an emoji
                    # (some emoji consists of multiple tokens)
                    # continue to generate more tokens to decode this emoji
                    continue
                text = text[len(history_text):]
                history_text += text
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "text": text,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_len
                    }
                }

    def _tokenize(self, text: str, *args) -> List[int]:
        """Tokenize a string.

        Args:
            text: The string to tokenize.

        Raises:
            RuntimeError: If the tokenization failed.

        Returns:
            A list of tokens.
        """
        warnings.warn("The parameter `add_bos` is unsupported, please use the default value.")
        return chatglm_tokenize(self.ctx, text)

    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize a list of tokens.

        Args:
            tokens: The list of tokens to detokenize.

        Returns:
            The detokenized string.
        """
        if isinstance(tokens, int):
            tokens = [tokens]
        return chatglm_detokenize(self.ctx, tokens)

    def forward(self,
                input_ids: List[int],
                do_sample: bool = True,
                top_k: int = 0,
                top_p: float = 0.7,
                temperature: float = 0.95,) -> int:
        return chatglm_forward(ctx=self.ctx,
                               input_ids=input_ids,
                               do_sample=do_sample,
                               top_k=top_k,
                               top_p=top_p,
                               temperature=temperature)

    def eos_token(self) -> int:
        return chatglm_eos_token(self.ctx)

    def _generate(
        self,
        tokens: Sequence[int],
        top_k: int = 0,
        top_p: float = 0.7,
        temp: float = 0.95,
        repeat_penalty: float = 1.1,
        reset: bool = True,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
    ) -> Generator[int, Optional[Sequence[int]], None]:
        """Create a generator of tokens from a prompt.

        Examples:
            >>> llm = ChatGLM(your_model_path)
            >>> tokens = llm._tokenize(b"Learning English is")
            >>> for token in llm._generate(tokens):
            >>>     print(llm.detokenize([token]).decode("utf-8", errors="ignore"))

        Args:
            tokens: The prompt tokens.

        Yields:
            The generated tokens.
        """
        # TODO: Some parameters are temporarily not supported
        # Unsupported parameters are checked in `_supported_generate`
        return self._supported_generate(tokens, top_k, top_p, temp, repeat_penalty, reset,
                                        frequency_penalty, presence_penalty, tfs_z, mirostat_mode,
                                        mirostat_tau, mirostat_eta)

    def _supported_generate(self, tokens: Sequence[int], top_k: int = 0, top_p: float = 0.7,
                            temp: float = 0.95, *args):
        # Check unsupporeted parameters
        unsupported_arg = ['repeat_penalty', 'reset', 'frequency_penalty', 'presence_penalty',
                           'tfs_z', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta']
        defult_value = {'repeat_penalty': 1.1, 'reset': True, 'frequency_penalty': 0.0,
                        'presence_penalty': 0.0, 'tfs_z': 1.0, 'mirostat_mode': 0,
                        'mirostat_tau': 5.0, 'mirostat_eta': 0.1}
        for index in range(len(args)):
            if args[index] != defult_value[unsupported_arg[index]]:
                warnings.warn(f"The parameter {unsupported_arg[index]} is temporarily "
                              "unsupported, please use the default value.")

        invalidInputError(self.ctx is not None, "The attribute `ctx` of `ChatGLM` object is None.")
        while True:
            token = self.forward(input_ids=tokens,
                                 top_k=top_k,
                                 top_p=top_p,
                                 temperature=temp)
            tokens_or_none = yield token
            tokens.append(token)
            if tokens_or_none is not None:
                tokens.extend(tokens_or_none)
