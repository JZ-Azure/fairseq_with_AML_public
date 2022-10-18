## Local Setup 
Install Azure Machine Learning Python SDK v1
```bash
pip install azureml-core
```
Install `ml` extension
```bash
az extension add -n ml
```

## Create AML workspace [Reference](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public)
```bash
az login
az account set --name Resource_Group

GROUP="azureml-examples"
LOCATION="southcentralus"
WORKSPACE="main"

# Create the resource group if it doesn't alreayd exist. 
az group create -n $GROUP -l $LOCATION

# Create AML workspace
az ml workspace create -n $WORKSPACE -g $GROUP -l $LOCATION
```

## Build the docker environment [Reference](https://learn.microsoft.com/en-us/azure/machine-learning/concept-environments)
```bash
cd environment
az ml environment create --file pytorch_env.yml
```

## Submit the job to AML
```bash
python aml_cluster.py
```

Sample output
```bash
2022-10-18 04:37:45 | INFO | fairseq.trainer | NOTE: gradient overflow detected, ignoring gradient, setting loss scale to: 64.0
2022-10-18 04:37:53 | INFO | fairseq.trainer | NOTE: gradient overflow detected, ignoring gradient, setting loss scale to: 32.0
2022-10-18 04:38:00 | INFO | fairseq.trainer | NOTE: gradient overflow detected, ignoring gradient, setting loss scale to: 16.0
2022-10-18 04:38:08 | INFO | fairseq.trainer | NOTE: gradient overflow detected, ignoring gradient, setting loss scale to: 8.0
2022-10-18 04:39:17 | INFO | train_inner | {"epoch": 1, "update": 0.027, "loss": "21.229", "moe_gate_loss": "21.6115", "overflow_expert1": "18.806", "overflow_expert2": "58.825", "entropy_gating": "1.964", "expert1_balance_top": "65.277", "expert1_balance_bottom": "2.345", "unused_expert1_count": "0.583", "expert2_balance_top": "50.332", "expert2_balance_bottom": "4.213", "unused_expert2_count": "0.332", "all_to_all_cpu_time_ms": "0", "all_to_all_cuda_time_ms": "0", "inner_loss": "20.918", "ppl": "1.98071e+06", "wps": "29046", "ups": "0.15", "wpb": "196608", "bsz": "192", "num_updates": "10", "lr": "7.33333e-06", "gnorm": "27.355", "loss_scale": "8", "train_wall": "409", "cuda_gb_allocated": "16.5", "cuda_gb_reserved": "30.1", "cuda_gb_free": "62.9", "wall": "410"}
2022-10-18 04:40:25 | INFO | train_inner | {"epoch": 1, "update": 0.046, "loss": "15.052", "moe_gate_loss": "18.1331", "overflow_expert1": "8.916", "overflow_expert2": "49.889", "entropy_gating": "2.028", "expert1_balance_top": "53.963", "expert1_balance_bottom": "4.017", "unused_expert1_count": "0.642", "expert2_balance_top": "42.744", "expert2_balance_bottom": "5.92", "unused_expert2_count": "0.484", "all_to_all_cpu_time_ms": "0", "all_to_all_cuda_time_ms": "0", "inner_loss": "14.79", "ppl": "28334.4", "wps": "29032.4", "ups": "0.15", "wpb": "196608", "bsz": "192", "num_updates": "20", "lr": "1.4e-05", "gnorm": "2.701", "loss_scale": "8", "train_wall": "68", "cuda_gb_allocated": "16.5", "cuda_gb_reserved": "30.1", "cuda_gb_free": "62.9", "wall": "478"}
2022-10-18 04:40:58 | INFO | fairseq_cli.train | Stopping training due to num_updates: 25 >= max_update: 25
2022-10-18 04:40:58 | INFO | fairseq.checkpoint_utils | Preparing to save checkpoint for epoch 1 @ 25 updates
2022-10-18 04:41:08 | INFO | fairseq.trainer | Saving checkpoint to ./checkpoint_last-rank-0-shard0.pt
2022-10-18 04:41:26 | INFO | fairseq.trainer | Finished saving checkpoint to ./checkpoint_last-rank-0-shard0.pt
2022-10-18 04:41:26 | INFO | fairseq.trainer | Saving checkpoint to ./checkpoint_last-shared-shard0.pt
2022-10-18 04:41:48 | INFO | fairseq.trainer | Finished saving checkpoint to ./checkpoint_last-shared-shard0.pt
2022-10-18 04:41:48 | INFO | fairseq.checkpoint_utils | Saved checkpoint ./checkpoint_last-rank-0-shard0.pt (epoch 1 @ 25 updates, score None) (writing took 49.164279436998186 seconds)
2022-10-18 04:41:48 | INFO | fairseq_cli.train | end of epoch 1 (average epoch stats below)
2022-10-18 04:41:48 | INFO | train | {"epoch": 1, "train_loss": "17.131", "train_moe_gate_loss": "19.2701", "train_overflow_expert1": "12.021", "train_overflow_expert2": "50.689", "train_entropy_gating": "2.006", "train_expert1_balance_top": "57.114", "train_expert1_balance_bottom": "4.243", "train_unused_expert1_count": "0.522", "train_expert2_balance_top": "44.746", "train_expert2_balance_bottom": "6.444", "train_unused_expert2_count": "0.357", "train_all_to_all_cpu_time_ms": "0", "train_all_to_all_cuda_time_ms": "0", "train_inner_loss": "16.853", "train_ppl": "118351", "train_wps": "22296.5", "train_ups": "0.11", "train_wpb": "196608", "train_bsz": "192", "train_num_updates": "25", "train_lr": "1.73333e-05", "train_gnorm": "12.484", "train_loss_scale": "8", "train_train_wall": "510", "train_cuda_gb_allocated": "16.5", "train_cuda_gb_reserved": "30.1", "train_cuda_gb_free": "62.9", "train_wall": "561"}
2022-10-18 04:41:48 | INFO | fairseq_cli.train | done training in 559.6 seconds
```
