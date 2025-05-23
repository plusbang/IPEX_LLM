# Yuan2
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Yuan2 models. For illustration purposes, we utilize the [IEITYuan/Yuan2-2B-hf](https://huggingface.co/IEITYuan/Yuan2-2B-hf) as a reference Yuan2 model.

## 0. Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

In addition, you need to modify some files in Yuan2-2B-hf folder, since Flash attention dependency is for CUDA usage and currently cannot be installed on Intel CPUs. To manually turn it off, please refer to [this issue](https://github.com/IEIT-Yuan/Yuan-2.0/issues/92). We also provide two modified files([config.json](yuan2-2B-instruct/config.json) and [yuan_hf_model.py](yuan2-2B-instruct/yuan_hf_model.py)), which can be used to replace the original content in config.json and yuan_hf_model.py.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for an Yuan2 model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[all] # install the latest bigdl-llm nightly build with 'all' option
pip install einops # additional package required for Yuan2 to conduct generation
pip install pandas # additional package required for Yuan2 to conduct generation
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Yuan2 model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'IEITYuan/Yuan2-2B-hf'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'IEITYuan/Yuan2-2B-hf'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `100`.

> **Note**: When loading the model in 4-bit, BigDL-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Yuan2 model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```powershell
python ./generate.py
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set BigDL-LLM env variables
source bigdl-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 2.3 Sample Output
#### [IEITYuan/Yuan2-2B-hf](https://huggingface.co/IEITYuan/Yuan2-2B-hf)
```log
Inference time: xxxx seconds
-------------------- Output --------------------
 
What is AI?
AI is what we call "Artificial Intelligence."<eod>
```