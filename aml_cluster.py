import os
import requests
import sys

# AzureML libraries
import azureml.core
from azureml.core import Dataset, Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import PyTorchConfiguration
from azureml.core.environment import DockerBuildContext

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)
#ws = Workspace.from_config()
ws = Workspace.get(name='main', subscription_id='f5a67d06-2d09-4090-91cc-e3298907a021', resource_group='JZ-azureml')
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')


#-------------------------------------------------------------------------------
# Prepare Compute Cluster
#-------------------------------------------------------------------------------
cluster_name = "JingchaoA100"

# Verify that the cluster doesn't exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_ND96amsr_A100_v4', min_nodes=0, max_nodes=2)
    
    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

#-------------------------------------------------------------------------------
# Setup training environment
#-------------------------------------------------------------------------------
fairseq_env = Environment.get(ws, name="fairseq", version="v10172022_1")
fairseq_env.python.user_managed_dependencies = True

#-------------------------------------------------------------------------------
# Training Settings and Arguments
#-------------------------------------------------------------------------------
SAVE_DIR = "/workspace/checkpoints"
NUM_GPUS_PER_NODE = 8
NUM_EXPERTS = 8
TOKENS_PER_SAMPLE = 1024
BATCH_SIZE = 6
MAX_UPDATE = 25
GRAD_ACC = 2

run_args = [
    '--save-dir', SAVE_DIR,
    '--ddp-backend', 'fully_sharded',
    '--memory-efficient-fp16', 
    '--checkpoint-activations',
    '--task', 'dummy_lm', 
    '--tokens-per-sample', TOKENS_PER_SAMPLE,
    '--arch', 'transformer_lm_gpt', 
    '--share-decoder-input-output-embed',
    '--decoder-layers', 32, 
    '--decoder-embed-dim', 4096, 
    '--decoder-ffn-embed-dim', 16384,
    '--decoder-attention-heads', 32,
    '--moe-expert-count', NUM_EXPERTS, 
    '--moe-freq', 2,
    '--moe-gating-use-fp32',
    '--moe-second-expert-policy', 'all',
    '--moe-normalize-expert-grad', 'sqrt_world_size',
    '--moe-eval-capacity-token-fraction', -1.0,
    '--max-sentences-valid', 1, 
    '--num-workers-valid', 0,
    '--criterion', 'moe_cross_entropy', 
    '--moe-gate-loss-wt', 0.01, 
    '--moe-gate-loss-combine-method', 'sum',
    '--optimizer', 'adam', 
    '--fp16-adam-stats', 
    #'--adam-betas', (0.9,0.98), 
    '--clip-norm', 0.0,
    '--lr', 0.0005, 
    '--warmup-updates', 750,
    '--dropout', 0.1, 
    '--attention-dropout', 0.1,
    '--batch-size', BATCH_SIZE, 
    '--update-freq', GRAD_ACC,
    '--max-update', MAX_UPDATE, 
    '--disable-validation',
    '--log-format', 'json', 
    '--log-interval', 10,
]

#-------------------------------------------------------------------------------
# Create ScriptRunConfig
#-------------------------------------------------------------------------------
distr_config = PyTorchConfiguration(node_count=2)
launch_cmd = "cd /workspace/fairseq && \
        python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 \
        --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
        train.py \
        --save-dir ./ \
        --ddp-backend fully_sharded --memory-efficient-fp16 --checkpoint-activations \
        --task dummy_lm --tokens-per-sample 1024 \
        --arch transformer_lm_gpt --share-decoder-input-output-embed \
        --decoder-layers 32 --decoder-embed-dim 4096 --decoder-ffn-embed-dim 16384 \
        --decoder-attention-heads 32 \
        --moe-expert-count 8 --moe-freq 2 \
        --moe-gating-use-fp32 --moe-second-expert-policy all \
        --moe-normalize-expert-grad sqrt_world_size \
        --moe-eval-capacity-token-fraction -1.0 \
        --max-sentences-valid 1 --num-workers-valid 0 \
        --criterion moe_cross_entropy --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum \
        --optimizer adam --fp16-adam-stats --adam-betas '(0.9,0.98)' --clip-norm 0.0 \
        --lr 0.0005 --warmup-updates 750 \
        --dropout 0.1 --attention-dropout 0.1 \
        --batch-size 6 --update-freq 2 \
        --max-update 25 --disable-validation \
        --log-format json --log-interval 10".split()

#launch_cmd = "echo $NODE_RANK && echo $MASTER_ADDR && echo $MASTER_PORT"

megatron_ds_src = ScriptRunConfig(
    source_directory='/home/jingchao/azureml-assets/jingchao_fairseq/src',
    #script='train.py',
    #arguments=run_args,
    command=launch_cmd,
    compute_target=compute_target,
    environment=fairseq_env,
    distributed_job_config=distr_config
    )

#-------------------------------------------------------------------------------
# Submit experiment
#-------------------------------------------------------------------------------
run = Experiment(ws, 'fairseq_PythonSDK').submit(megatron_ds_src)
run.wait_for_completion(show_output=True)
