PROJECTS_DIR=$HOME/projects
SLIME_DIR=$PROJECTS_DIR/slime
cd $SLIME_DIR

if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Activating..."
else
    echo "Creating virtual environment..."
    uv venv -p 3.12
fi
source .venv/bin/activate

uv pip install nvtx cuda-python==13.1.0
uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129
uv pip install torch-c-dlpack-ext
cd $PROJECTS_DIR

if [ ! -d "sglang" ]; then
    git clone https://github.com/TideDra/sglang.git
fi
cd sglang
#checkout to branch "dev"
git checkout slime_0.5.9
uv pip install -e "python"
wget https://github.com/TideDra/sglang/releases/download/slime_0.5.7/sglang_router-0.3.0-cp38-abi3-manylinux_2_34_x86_64.whl
uv pip install sglang_router-0.3.0-cp38-abi3-manylinux_2_34_x86_64.whl
cd $PROJECTS_DIR

if [ ! -d "inference" ]; then
    git clone https://github.com/t2vg/inference.git
fi

cd inference/backend/agent_core/
uv pip install -e .
cd $PROJECTS_DIR
uv pip install cmake ninja wheel

MAX_JOBS=64 uv pip -v install flash-attn==2.7.4.post1 --no-build-isolation

uv pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

export CUDNN_PATH=$SLIME_DIR/.venv/lib/python3.12/site-packages/nvidia/cudnn
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH
export CPATH=$CUDNN_PATH/include:$CPATH
uv pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"

APEX_CPP_EXT=1 APEX_CUDA_EXT=1 NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 uv pip install -v --no-build-isolation git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

uv pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --force-reinstall

uv pip install poetry pybind11

uv pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation

uv pip install nvidia-modelopt --no-build-isolation

cd $PROJECTS_DIR
if [ ! -d "Megatron-LM" ]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi

cd Megatron-LM
git checkout 3714d81d418c9f1bca4594fc35f9e8289f652862
uv pip install -e .
git apply $SLIME_DIR/docker/patch/latest/megatron.patch

cd $SLIME_DIR

uv pip install -e .

uv pip install numpy==1.26.4

uv pip install nvidia-cudnn-cu12==9.16.0.29