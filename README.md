# MobileLLM-ZeroShot-Reasoning

## Overview
This repository contains the implementation of fine-tuning Facebook's MobileLLM-125M model for zero-shot reasoning tasks using the HellaSwag dataset. The project demonstrates how to adapt lightweight language models for edge devices while maintaining reasonable performance on complex reasoning tasks.


## Key Features
- Implementation of MobileLLM-125M fine-tuning pipeline
- Adaptation for commonsense reasoning using HellaSwag dataset
- Performance evaluation metrics and comparison with baseline
- Deployment-ready code with FastAPI integration
- Optimized for resource-constrained environments

## Model Architecture
MobileLLM-125M is a lightweight transformer-based model designed for on-device applications with:
- 125 million parameters
- Transformer architecture with optimizations for efficiency
- Reduced memory footprint and inference latency
- Techniques applied: weight pruning, quantization, and knowledge distillation

## Dataset: HellaSwag
HellaSwag is a challenging dataset for commonsense reasoning and natural language inference:
- Multiple-choice question-answering format
- 39,905 training samples, 10,042 validation samples, 10,003 test samples
- Each example contains a context and four possible completions
- Requires nuanced understanding of context and logical reasoning

## Implementation Details

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- Transformers 4.18.0+
- Accelerate 0.5.0+
- 4 A100 GPUs (for full training) or at least 16GB VRAM for single GPU training

### Installation
```bash
git clone https://github.com/YourUsername/MobileLLM-ZeroShot-Reasoning.git
cd MobileLLM-ZeroShot-Reasoning
pip install -r requirements.txt
```

### Data Preprocessing
The preprocessing pipeline includes:
1. Loading the HellaSwag dataset
2. Constructing input sequences by concatenating context with each possible ending
3. Tokenizing sequences with a maximum length of 512 tokens
4. Creating training features: input IDs, attention masks, and labels

```python
# Sample preprocessing code
def preprocess_function(examples):
    # Concatenate context with endings
    inputs = []
    for ctx, endings in zip(examples["ctx"], examples["endings"]):
        for ending in endings:
            inputs.append(f"{ctx} {ending}")
    
    # Tokenize inputs
    tokenized_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Prepare labels for causal language modeling
    labels = tokenized_inputs["input_ids"].clone()
    
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels
    }
```

### Training Pipeline
Our training implementation uses Hugging Face's Accelerate library for distributed training:

1. **Model Initialization**:
   - Load pretrained MobileLLM-125M model
   - Configure tokenizer with appropriate padding token

2. **Optimization Setup**:
   - Learning rate: 1e-5
   - Optimizer: AdamW with weight decay
   - Scheduler: Cosine learning rate with warmup

3. **Training Loop**:
   - Epochs: 3
   - Batch size: 16 per GPU
   - Gradient accumulation for stable training

### Evaluation
We evaluate model performance using multiple metrics:
- Validation loss: 2.068
- Validation accuracy: 62.8%
- Validation perplexity: 7.91

Our fine-tuned model achieves a validation accuracy of 62.8%, which is close to the baseline performance of 65.3% reported for MobileLLM-125M on HellaSwag.

## Results
Fine-tuning progress across epochs:
- Epoch 1: Loss 0.9609
- Epoch 2: Loss 0.8557
- Epoch 3: Loss 0.8034
- Epoch 4: Loss 0.7775
- Epoch 5: Loss 0.7680

Final validation loss: 0.8131

## Deployment
The repository includes deployment code using FastAPI and Docker:

1. **FastAPI Implementation**:
   - RESTful API for text generation
   - Efficient request handling

2. **Docker Container**:
   - Ensures portability and scalability
   - Contains all dependencies and model weights

3. **Optimization Techniques**:
   - Caching for repeated requests
   - Dynamic batching for throughput
   - GPU utilization management

## Usage

### Training
```bash
# Single GPU
python train.py --dataset hellaswag --model facebook/MobileLLM-125M --output_dir ./fine_tuned_model

# Multi-GPU
accelerate launch train.py --dataset hellaswag --model facebook/MobileLLM-125M --output_dir ./fine_tuned_model
```

### Inference
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model
model_path = "./fine_tuned_model"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Generate text
input_text = "The chef is preparing a meal. He takes out a knife and"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### API
After starting the FastAPI server:
```bash
curl -X POST "http://localhost:8000/generate/" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "The chef is preparing a meal. He takes out a knife and", "max_length": 100}'
```

## Future Work
- Model optimization using pruning and quantization techniques
- Exploration of automated scaling for high-throughput requests
- Integration with edge computing frameworks for mobile deployment
- Further fine-tuning on domain-specific tasks

## References
1. Vaswani, A., et al. (2017). "Attention is all you need." NeurIPS.
2. Wolf, T., et al. (2020). "HuggingFace's Transformers: State-of-the-art Natural Language Processing."
3. Zellers, R., et al. (2019). "HellaSwag: Can a Machine Really Finish Your Sentence?" ACL.
4. Loshchilov, I., & Hutter, F. (2017). "Decoupled Weight Decay Regularization." ICLR 2018.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use this code or model in your research, please cite:
```
@misc{ela2023mobilellm,
  author = {Ela, Saiprudvi},
  title = {MobileLLM-ZeroShot-Reasoning},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/YourUsername/MobileLLM-ZeroShot-Reasoning}}
}
```
