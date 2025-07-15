# MedCoLLM: Collaboration Between Physicians and Large Language Models in Cancer Pain Medication Decision-Making


##  Lora Configuration

- **Directory**: `MedCoLLM/LLaMA-Factory/examples/train_nscale`  
  Contains the configuration files required for LoRA fine-tuning.

## Distributed Training Configuration

- **File**: `MedCoLLM/examples/deepspeed/ds_z3_config.json`  
  Uses ZeRO Stage 3 configuration for distributed training with DeepSpeed.

##  Sample Data

- **File**: `MedCoLLM/DATA/Data(sample).csv`  
  A sample data file for testing or training purposes.


## Simple Test Lora

- **Path**: `MedCoLLM/LLaMA-Factory/tests/test1.py`  
  This script is used for simple model loading and inference tests.
