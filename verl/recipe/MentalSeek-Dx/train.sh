#!/bin/bash

# =============================================================================
# Progressive HDR (Progressive Hypothetico-Deductive Reasoning) Training Script
# For MentalSeek-Dx Reinforcement Learning Training
# =============================================================================

# Project and experiment name configuration
project_name=MentalSeek-Dx
exp_name=MentalSeek-Dx-7B

#exp_name=MentalSeek-Dx-7B or MentalSeek-Dx-14B


# Disable tokenizer parallelism to avoid multiprocessing conflicts
export TOKENIZERS_PARALLELISM=false

# Allow vLLM to process long sequences
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Set Ray temporary directory to a location with sufficient space (avoid system disk space issues)
export RAY_TMPDIR=./verl/ray_tmp
mkdir -p $RAY_TMPDIR

# Allow Ray to use external object storage (improves large model training efficiency)
export RAY_OBJECT_STORE_ALLOW_EXTERNAL_STORAGE=1

# Disable Ray debug mode (improves training performance)
export RAY_DEBUG=0
unset RAY_DEBUG_POST_MORTEM

# =============================================================================
# GRPO core algorithm parameter configuration
# =============================================================================
# Use GRPO (Group Relative Policy Optimization) algorithm
# GRPO optimizes strategy by within-group relative comparison, suitable for dialogue generation tasks
adv_estimator=grpo

# =============================================================================
# Data loading and processing parameters
# =============================================================================
# Training batch size (number of samples processed per step)
train_batch_size=64

# Maximum prompt length
max_prompt_length=2048

# Maximum response length
max_response_length=2048

# Filter overlong prompts
filter_overlong_prompts=True

# Handling for overlong text: error
truncation='error'

# Whether to shuffle data order
shuffle=True

# =============================================================================
# Model and data path configuration
# =============================================================================
# Pretrained model path
MODEL_PATH=path/to/MentalSeek-SFT-model

# Training results save path
CKPTS_DIR=path/to/MentalSeek-Dx-7B

# Training data path
TRAIN_FILE=path/to/train-data

# Validation data path
TEST_FILE=path/to/MentalDX.parquet

# =============================================================================
# LoRA (Low-Rank Adaptation) parameter configuration
# =============================================================================
# LoRA rank (controls number of tunable parameters; 64 is suitable for 7B models)
lora_rank=64

# LoRA scaling factor (usually set to half of rank)
lora_alpha=32

# =============================================================================
# PPO core training parameters
# =============================================================================
# Actor learning rate (3e-6 is suitable for medical domain fine-tuning)
actor_lr=3e-6

# PPO mini-batch size (must be divisible by ppo_micro_batch_size_per_gpu)
ppo_mini_batch_size=64

# Per-GPU micro-batch size (adjust based on GPU memory)
ppo_micro_batch_size_per_gpu=8

# =============================================================================
# KL divergence constraint parameters (prevent model from deviating far from reference policy)
# =============================================================================
# Enable KL divergence loss
use_kl_loss=True

# KL divergence coefficient (0.001 is a weak constraint, good for medical dialogue)
kl_loss_coef=0.001

# KL type (low_var_kl reduces variance and improves stability)
kl_loss_type=low_var_kl

# =============================================================================
# Entropy regularization parameters (controls output diversity)
# =============================================================================
# Entropy coefficient (0 means no exploration encouragement, good for medical task consistency)
entropy_coeff=0

# =============================================================================
# Memory and performance optimization parameters
# =============================================================================
# Enable gradient checkpointing (reduce memory usage, increase computation time)
enable_gradient_checkpointing=True

# FSDP param offloading (False keeps parameters on GPU, improves speed)
param_offload=False

# Optimizer state offloading (False keeps optimizer state on GPU)
optimizer_offload=False

# =============================================================================
# Rollout generation parameters (model response generation config)
# =============================================================================
# Rollout log-prob micro-batch size per GPU
log_prob_micro_batch_size_per_gpu=64

# Tensor model parallel size (1 is suitable for 2-GPU parallelism)
tensor_model_parallel_size=1

# vLLM GPU memory utilization (1.0 means use full memory for vLLM)
gpu_memory_utilization=1.0

# Number of responses generated per prompt (8 candidate responses)
n_resp_per_prompt=8

# Model load format (safetensors is safer)
load_format=safetensors

# Enable layered loading (improves large model load efficiency)
layered_summon=True

# =============================================================================
# Algorithm control parameters
# =============================================================================
# Whether to use KL divergence in reward (False uses separate KL loss)
use_kl_in_reward=False

# Critic network warmup steps (0 means enable Critic immediately)
critic_warmup=10

# =============================================================================
# CUDA environment configuration
# =============================================================================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# =============================================================================
# Trainer general parameters
# =============================================================================
# Number of GPUs per node (8-GPU training)
n_gpus_per_node=8

# Number of nodes (1-node training)
nnodes=1

# Model save frequency (save every 40 steps)
save_freq=40

# Validation frequency (validate every 100 steps)
test_freq=100

total_epochs=5
total_training_steps=200
# Whether to validate before training (False means directly start training)
val_before_train=False

# =============================================================================
# Generation sampling parameters (controls output randomness)
# =============================================================================
# Sampling temperature (1.0 keeps original probability distribution)
temperature=1.0

# Top-p sampling threshold (0.99 considers nearly all vocabulary)
top_p=0.99

# Top-k sampling (-1 means no restriction, use all vocabulary)
top_k=-1

# python3 -m debugpy --wait-for-client --listen 5678 verl/trainer/main_ppo.py \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    trainer.val_before_train=$val_before_train \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=$filter_overlong_prompts \
    data.truncation=$truncation \
    data.shuffle=$shuffle \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=$lora_rank \
    actor_rollout_ref.model.lora_alpha=$lora_alpha \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=$enable_gradient_checkpointing \
    actor_rollout_ref.actor.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.load_format=$load_format \
    actor_rollout_ref.rollout.layered_summon=$layered_summon \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    custom_reward_function.path="MentalSeek-Dx/verl/recipe/MentalSeek-Dx/compute_score.py" \
    custom_reward_function.name="reward_function" \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger='["console","wandb"]' \
    trainer.wandb_log_model_outputs=False \
    trainer.wandb_log_metrics=True \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    trainer.total_training_steps=$total_training_steps \
    trainer.default_local_dir=$CKPTS_DIR \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.val_kwargs.temperature=$temperature \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=$top_k \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1