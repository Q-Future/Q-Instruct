mkdir -p ./checkpoints/llava-v1.5-13b-pretrain/
mkdir -p ./checkpoints/llava-v1.5-7b-pretrain/

wget -P ./checkpoints/llava-v1.5-13b-pretrain/ https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5/resolve/main/mm_projector.bin
wget -P ./checkpoints/llava-v1.5-7b-pretrain/ https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/resolve/main/mm_projector.bin
