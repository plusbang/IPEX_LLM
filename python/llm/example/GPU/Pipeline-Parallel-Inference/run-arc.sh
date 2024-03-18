source /opt/intel/oneapi/setvars.sh
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so:${LD_PRELOAD}
python qwen2_generate.py --low-bit 'sym_int8'
# python qwen2_generate.py --low-bit 'fp8_e5m2'
