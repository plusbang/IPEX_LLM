source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:6
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export ENABLE_SDP_FUSION=1
export SYCL_CACHE_PERSISTENT=1
KERNEL_VERSION=$(uname -r)
if [[ $KERNEL_VERSION != *"6.5"* ]]; then
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
fi
export BIGDL_QUANTIZE_KV_CACHE=1
python run.py # make sure config YAML file