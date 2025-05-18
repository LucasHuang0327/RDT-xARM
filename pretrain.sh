export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-pretrain-1b"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="/path/to/cutlass"

export WANDB_PROJECT="robotics_diffusion_transformer"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...

deepspeed --hostfile=hostfile.txt main.py \     # 指定分布式訓練主機列表 
    --deepspeed="./configs/zero2.json" \        # deepspeed的配置文件, 定義分布式訓練和內存優化策略（如 ZeRO 優化器）。
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \     # 預訓練的文本編碼器名稱
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \ # 預訓練的視覺編碼器名稱
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=32 \         # 訓練的batch size
    --sample_batch_size=64 \        # 這是生成的batch size (可以理解為action chunk嗎? )
    --max_train_steps=1000000 \     # 訓練的最大步數
    --checkpointing_period=1000 \   # 每1000步保存一次檢查點
    --sample_period=500 \           # 每500步生成一次樣本
    --checkpoints_total_limit=40 \  # 最多保存40個檢查點
    --lr_scheduler="constant" \     # 學習率調度器類型為constant
    --learning_rate=1e-4 \          # 初始學習率
    --mixed_precision="bf16" \      # 訓練精度使用bf16
    --dataloader_num_workers=8 \    # 數據加載器的工作進程數量
    --dataset_type="pretrain" \
    --report_to=wandb

    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-1000" \
