python -m scripts.xArm_inference \
    --use_actions_interpolation \
    --pretrained_model_name_or_path="checkpoints/rdt-coke-finetune-1b/checkpoint-5000/pytorch_model/mp_rank_00_model_states.pt" \
    --lang_embeddings_path="data/empty_lang_embed.pt" \
    --ctrl_freq=25    # your control frequency

 # your finetuned checkpoint: e.g., checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>, checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>/pytorch_model/mp_rank_00_model_states.pt,