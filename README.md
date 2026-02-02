# MentalSeek-Dx: Towards Progressive Hypothetico-Deductive Reasoning for Real-world Psychiatric Diagnosis

<div align="center">

**A Medical-Specialized Large Language Model for Clinical Psychiatric Diagnosis**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-VERL-green.svg)](https://github.com/volcengine/verl)

</div>

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Component Overview](#-component-overview)
- [Environment Setup](#-environment-setup)
- [Model Weights](#-model-weights)
- [Quick Start](#-quick-start)
- [Model Training](#-model-training)
- [Evaluation](#-evaluation)
- [Citation](#-citation)

---

## 🎯 Introduction

Mental health disorders represent a growing global public health concern, with increasing demand for accurate and accessible diagnostic tools. While large language models (LLMs) show promise for psychiatric assessment, existing benchmarks often lack the real-world complexity and detailed diagnostic supervision necessary for clinical applicability.

### Problem Statement

Current LLM-based diagnostic systems face a fundamental **paradigm misalignment**: while many models demonstrate strong performance in broad diagnostic categorization, they struggle with fine-grained, disorder-level diagnosis. This reveals a critical gap between pattern recognition capabilities and rigorous clinical reasoning processes.

### Our Solution

**MentalSeek-Dx** is a medical-specialized LLM designed to bridge this gap by internalizing clinical hypothetico-deductive reasoning (HDR). The model is trained using:

- **Supervised trajectory construction** to capture clinical reasoning patterns
- **Curriculum-based reinforcement learning** to simulate psychiatrist diagnostic processes
- **Progressive reasoning mechanisms** that align with real-world clinical workflows

### MentalDx Bench

We introduce **MentalDx Bench** — the first comprehensive benchmark targeting disorder-level psychiatric diagnosis in real clinical settings:

- **712 de-identified electronic health records** (EHRs)
- **Board-certified psychiatrist annotations** using ICD-11 guidelines
- **76 disorders** across **16 diagnostic categories**
- **Disorder-level granularity** for fine-grained evaluation

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🏆 **SOTA Performance** | State-of-the-art results on MentalDx Bench with only 14B parameters |
| 🧠 **Clinical Reasoning** | Deep integration of hypothetico-deductive reasoning, moving beyond pattern-based diagnosis |
| 🎯 **Disorder-Level Precision** | Fine-grained diagnostic capabilities at the disorder level, not just category level |
| 📊 **Comprehensive Benchmark** | First benchmark with real-world EHRs and disorder-level annotations |
| 🔬 **Trustworthy Foundation** | Establishes a practical, evidence-based foundation for psychiatric diagnosis with LLMs |

---

## 📁 Project Structure

```
MentalSeek-Dx/
├── README.md                                    # Main project documentation
├── requirements.txt                             # Python dependency list
├── .gitignore                                   # Git ignore configuration
├── run_MentalSeek-Dx.py                         # Main execution script
│
├── Reasoning_Trajectory_Building/               # Module for reasoning trajectory construction
│   └── prompts.py                               # Prompt templates and reasoning trajectory logic
│
├── Structured_Psychiatric_Knowledge_Base/       # Structured psychiatric knowledge base
│   ├── Criteria_part.jsonl                      # Diagnostic criteria partial data
│   └── Diagnostic_category_part.jsonl           # Diagnostic category partial data
│
├── MentalDx-Bench/                             # Benchmark dataset and evaluation tools
│   ├── Ground_Truth.jsonl                       # Ground-truth label data
│   ├── MentalDx-Bench.jsonl                     # Benchmark dataset
│   └── src/                                     # Evaluation source code
│       ├── extract_and_evaluate.py              # Results extraction and evaluation script
│       └── statistics_category_accuracy.py      # Category accuracy statistics script
│
├── Models/                                      # Model configurations and documentation
│   ├── MentalSeek-Dx-14B/
│   │   └── models.md                            # Documentation for 14B model
│   └── MentalSeek-Dx-7B/
│       └── models.md                            # Documentation for 7B model
│
├── eval_results/                                # Evaluation results directory
│   ├── extracted_result/                        # Extracted evaluation results
│   │   ├── MentalSeek-Dx-14B.jsonl             # 14B model evaluation results
│   │   └── MentalSeek-Dx-7B.jsonl              # 7B model evaluation results
│   ├── prediction_output/                       # Model prediction outputs
│   │   ├── MentalSeek-Dx-14B.jsonl             # 14B model predictions
│   │   └── MentalSeek-Dx-7B.jsonl              # 7B model predictions
│   └── result_statistics.txt                    # Aggregated result statistics
│
└── verl/                                        # VERL framework (third-party reinforcement learning toolkit)
    ├── verl/                                    # VERL core implementation
    ├── examples/                                # Example scripts and configurations
    ├── docs/                                    # Documentation
    ├── docker/                                  # Docker configurations
    ├── scripts/                                 # Utility scripts
    ├── tests/                                   # Test suites
    ├── recipe/                                  # Training recipes and configurations
    │   └── MentalSeek-Dx/                       # MentalSeek-Dx specific configuration
    ├── requirements.txt                         # VERL dependencies
    ├── setup.py                                 # Installation script
    └── README.md                                # VERL framework documentation
```

---

## 🧩 Component Overview

### Core Modules

| Module | Description |
|--------|-------------|
| **`run_MentalSeek-Dx.py`** | Main entry script for running the MentalSeek-Dx model inference and evaluation |
| **`Reasoning_Trajectory_Building/`** | Contains the essential logic for constructing reasoning trajectories and related prompt templates that simulate clinical diagnostic processes |
| **`Structured_Psychiatric_Knowledge_Base/`** | Stores structured psychiatric knowledge in JSONL format, including diagnostic criteria and category information |

### Evaluation Modules

| Component | Description |
|-----------|-------------|
| **`MentalDx-Bench/`** | Comprehensive benchmark dataset and evaluation utilities |
| `Ground_Truth.jsonl` | Gold-standard diagnostic labels annotated by board-certified psychiatrists |
| `MentalDx-Bench.jsonl` | Full benchmark dataset containing 712 de-identified electronic health records |
| `src/extract_and_evaluate.py` | Script for extracting model predictions and computing evaluation metrics |
| `src/statistics_category_accuracy.py` | Script for computing category-level and disorder-level accuracy statistics |

### Results Output

| Directory | Contents |
|-----------|----------|
| **`eval_results/extracted_result/`** | Extracted evaluation results in JSONL format, including diagnostic predictions and ground truth comparisons |
| **`eval_results/prediction_output/`** | Raw model prediction outputs before post-processing |
| **`eval_results/result_statistics.txt`** | Aggregated statistical summaries including accuracy metrics at category and disorder levels |

### Model Configurations

| Model | Description |
|-------|-------------|
| **MentalSeek-Dx-14B** | 14-billion parameter model with state-of-the-art performance on MentalDx Bench |
| **MentalSeek-Dx-7B** | 7-billion parameter model offering a more efficient alternative with competitive performance |

### Third-party Framework

**`verl/`** - VERL (Versatile Reinforcement Learning) Framework

- Provides the foundational reinforcement learning framework for training MentalSeek-Dx
- Includes trainers, worker nodes, utility functions, and comprehensive implementations
- Supports multiple training algorithms (e.g., PPO, GRPO, GPG, etc.)
- For detailed documentation, see: [VERL Repository](https://github.com/volcengine/verl)

---

## 🚀 Environment Setup

### Prerequisites

- **Python**: 3.11
- **CUDA**: Compatible GPU with CUDA support (for model inference and training)
- **Conda**: For environment management (recommended)

### Installation Steps

#### Step 1: Create Python Environment

```bash
# Create a new conda environment with Python 3.11
conda create -n mentalseek-dx python=3.11 -y
conda activate mentalseek-dx
```

#### Step 2: Install VERL Framework

The project uses VERL (Versatile Reinforcement Learning) framework for training. Please refer to the official VERL documentation for detailed installation instructions:

- **VERL Repository**: https://github.com/volcengine/verl
- **VERL Documentation**: See `verl/README.md` in this repository

> **Note**: VERL installation may require additional dependencies based on your hardware configuration (CUDA, NPU, etc.). Please follow the official VERL setup guide.

#### Step 3: Install Project Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

#### Step 4: Verify Installation

```bash
# Verify Python version
python --version  # Should output Python 3.11.x

# Verify key dependencies
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from modelscope import AutoModelForCausalLM; print('ModelScope installed successfully')"
```

---

## 📦 Model Weights

| Model Name | Parameters | Availability | Notes |
|------------|------------|---------------|-------|
| **MentalSeek-Dx-7B** | 7B | [https://huggingface.co/SAMLL5AAAAA/MentalSeek-Dx-7B](https://huggingface.co/SAMLL5AAAAA/MentalSeek-Dx-7B) | The MentalSeek-Dx-7B is already visible! |
| **MentalSeek-Dx-14B** | 14B | 🔒 Coming Soon | Model weights will be made publicly available upon paper acceptance |

### Benchmark Data

The MentalDx Bench dataset is included in [https://huggingface.co/datasets/SAMLL5AAAAA/MentalDx-Bench](https://huggingface.co/datasets/SAMLL5AAAAA/MentalDx-Bench):

- **Input File**: `MentalDx-Bench/MentalDx-Bench.jsonl`
- **Ground Truth Labels**: `MentalDx-Bench/Ground_Truth.jsonl`

> **Note**: The dataset contains 712 de-identified electronic health records. Please ensure proper data handling and compliance with relevant regulations when using this data.

---

## 🚀 Quick Start

### Inference Example

To perform inference with MentalSeek-Dx, use the following code:

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

# Initialize model and tokenizer
model_name = "path/to/MentalSeek-Dx-7B"  # or MentalSeek-Dx-14B

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Construct diagnostic prompt
prompt = """
You are a psychiatric diagnosis expert. I will provide you with the patient's medical records. 
Please analyze the medical record information I provide and give the corresponding diagnosis results, 
including the category, specific disorder, and corresponding ICD-11 code (must be precise to four digits).

## Diagnostic Category Reference: 
["Neurodevelopmental disorders", "Schizophrenia or other primary psychotic disorders", 
"Catatonia", "Mood disorders", "Anxiety or fear-related disorders", 
"Obsessive-compulsive or related disorders", "Disorders specifically associated with stress", 
"Dissociative disorders", "Feeding or eating disorders", "Elimination disorders", 
"Disorders of bodily distress or bodily experience", 
"Disorders due to substance use or addictive behaviours", 
"Impulse control disorders", "Disruptive behaviour or dissocial disorders", 
"Personality disorders and related traits", "Paraphilic disorders", 
"Neurocognitive disorders", 
"Mental or behavioural disorders associated with pregnancy, childbirth or the puerperium", 
"Factitious disorders"]

## Specific Disorders and Corresponding ICD-11 Code Reference:
["6A20 Schizophrenia", "6B00 Generalised anxiety disorder", "6A72 Dysthymic disorder", 
"6E20 Mental or behavioural disorders associated with pregnancy, childbirth or the puerperium, without psychotic symptoms", 
"6A61 Bipolar type II disorder", "6A00 Disorders of intellectual development", 
"6D70 Delirium", "6E21 Mental or behavioural disorders associated with pregnancy, childbirth or the puerperium, with psychotic symptoms", 
"6B20 Obsessive-compulsive disorder", "6A40 Catatonia associated with another mental disorder", 
"6D72 Amnestic disorder", "6A62 Cyclothymic disorder", "6A23 Acute and transient psychotic disorder", 
"6B43 Adjustment disorder", "6A21 Schizoaffective disorder", "6A70 Single episode depressive disorder", 
"6A73 Mixed depressive and anxiety disorder", "6A71 Recurrent depressive disorder", 
"6A24 Delusional disorder", "6A60 Bipolar type I disorder", "6C20 Bodily distress disorder", 
"6A22 Schizotypal disorder", "6A41 Catatonia induced by substances or medications", 
"6D71 Mild neurocognitive disorder", "6D85 Dementia due to diseases classified elsewhere", 
"6C4G Disorders due to use of unknown or unspecified psychoactive substances", 
"6C4E Disorders due to use of other specified psychoactive substances, including medications", 
"6C46 Disorders due to use of stimulants including amphetamines, methamphetamine or methcathinone", 
"6B04 Social anxiety disorder", "6C40 Disorders due to use of alcohol", "6D10 Personality disorder", 
"6A02 Autism spectrum disorder", "6B02 Agoraphobia", "6B60 Dissociative neurological symptom disorder", 
"6C91 Conduct-dissocial disorder", "6C90 Oppositional defiant disorder", 
"6B83 Avoidant-restrictive food intake disorder", "6B40 Post traumatic stress disorder", 
"6C4F Disorders due to use of multiple specified psychoactive substances, including medications", 
"6D84 Dementia due to psychoactive substances including medications", "6B63 Possession trance disorder", 
"6B03 Specific phobia", "6B41 Complex post traumatic stress disorder", 
"6A03 Developmental learning disorder", "6C51 Gaming disorder", 
"6D86 Behavioural or psychological disturbances in dementia", "6A06 Stereotyped movement disorder", 
"6D83 Frontotemporal dementia", "6B80 Anorexia Nervosa", "6B23 Hypochondriasis", 
"6C49 Disorders due to use of hallucinogens", "6B81 Bulimia Nervosa", 
"6A05 Attention deficit hyperactivity disorder", "6B62 Trance disorder", "6B84 Pica", 
"6B21 Body dysmorphic disorder", "6B05 Separation anxiety disorder", 
"6B64 Dissociative identity disorder", "6B61 Dissociative amnesia", 
"6A04 Developmental motor coordination disorder", 
"6C4D Disorders due to use of dissociative drugs including ketamine and phencyclidine [PCP]", 
"6B66 Depersonalization-derealization disorder", 
"6B25 Body-focused repetitive behaviour disorders", "6B42 Prolonged grief disorder", 
"6B82 Binge eating disorder", "6C44 Disorders due to use of sedatives, hypnotics or anxiolytics", 
"6B22 Olfactory reference disorder", "6C01 Encopresis", 
"6D81 Dementia due to cerebrovascular disease", "6B06 Selective mutism", 
"6C50 Gambling disorder", "6B24 Hoarding disorder", "6C41 Disorders due to use of cannabis", 
"6B85 Rumination-regurgitation disorder", "6C47 Disorders due to use of synthetic cathinones"]

## Output Format:
You need to wrap your diagnostic process with <think></think> placeholders and wrap the final 
diagnosis result with <answer></answer> placeholders. The diagnosis result should be output in 
JSON format, where:
- "diagnostic_category": Select the most likely category from the diagnostic categories listed above;
- "candidate_disorders": A list of candidate disorders you consider as suspected, only include the codes of candidate disorders;
- "disorder_code": The final confirmed disorder code;
- "specific_disorder": The final confirmed disorder name, according to the official ICD-11 name;

## Output Example:
-------------
<think>
[Your reasoning process here]
</think>
<answer>
{{
    "diagnostic_category": "Disorders specifically associated with stress",
    "candidate_disorders": ["6B43", "6B40"],
    "disorder_code": "6B43",
    "specific_disorder": "Adjustment disorder"
}}
</answer>

## Test Medical Record: {MEDICAL_RECORD}
"""

# Prepare messages
messages = [
    {"role": "system", "content": "You are MentalSeek-Dx, a psychiatric diagnosis expert"},
    {"role": "user", "content": prompt.format(MEDICAL_RECORD="[Your medical record here]")}
]

# Apply chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize and generate
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048,
    temperature=0.7,
    do_sample=True
)

# Decode response
generated_ids = [
    output_ids[len(input_ids):] 
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### Running the Main Script

You can also use the provided main script:

```bash
python run_MentalSeek-Dx.py --model_path path/to/MentalSeek-Dx-7B --input_file path/to/MentalDx-Bench.jsonl
```

---

## 🎓 Model Training

MentalSeek-Dx is trained using a two-stage approach:

### Stage 1: Cold Start Data Preparation

**Location**: `Reasoning_Trajectory_Building/`

This stage involves constructing supervised reasoning trajectories that capture the clinical diagnostic process. The trajectories are designed to:

- Simulate psychiatrist reasoning patterns
- Incorporate structured psychiatric knowledge
- Generate high-quality training data for the next stage

**Key Components**:
- `prompts.py`: Contains prompt templates and reasoning trajectory construction logic

### Stage 2: Progressive HDR Training

**Location**: `verl/recipe/MentalSeek-Dx/train.sh`

This stage uses curriculum-based reinforcement learning to train the model progressively:

- **Initial Phase**: Supervised fine-tuning on constructed trajectories
- **Progressive Phase**: Reinforcement learning with curriculum-based difficulty scaling
- **Final Phase**: Fine-tuning on high-quality diagnostic examples

**Training Configuration**:
- Framework: VERL (Versatile Reinforcement Learning)
- Algorithm: Progressive Hypothetico-Deductive Reasoning (HDR)
- Training Script: `verl/recipe/MentalSeek-Dx/train.sh`

> **Note**: For detailed training instructions and hyperparameter configurations, please refer to the training script and VERL documentation.

---

## 📊 Evaluation

### Running Evaluation

To evaluate MentalSeek-Dx on the MentalDx Bench dataset:

```bash
# Navigate to the evaluation directory
cd MentalDx-Bench/src

# Run the evaluation script
python extract_and_evaluate.py \
    --model_path path/to/MentalSeek-Dx-7B \
    --benchmark_file ../MentalDx-Bench.jsonl \
    --ground_truth_file ../Ground_Truth.jsonl \
    --output_dir ../../eval_results
```

### Evaluation Metrics

The evaluation script computes:

- **Category-Level Accuracy**: Accuracy at the diagnostic category level (16 categories)
- **Disorder-Level Accuracy**: Accuracy at the specific disorder level (76 disorders)
- **Jiont Accuracy**:  Accuracy at both the diagnostic category and the specific disorder level

### Viewing Results

Results are saved in the `eval_results/` directory:

- `extracted_result/`: Detailed evaluation results in JSONL format
- `prediction_output/`: Raw model predictions
- `result_statistics.txt`: Aggregated statistics and metrics

To view detailed statistics:

```bash
cat eval_results/result_statistics.txt
```

Or use the statistics script:

```bash
python MentalDx-Bench/src/statistics_category_accuracy.py \
    --result_file eval_results/extracted_result/MentalSeek-Dx-7B.jsonl
```

---

## 📝 Citation

If you use MentalSeek-Dx or MentalDx Bench in your research, please cite our paper:

```bibtex
@article{mentalseek-dx2025,
  title={MentalSeek-Dx: Towards Progressive Hypothetico-Deductive Reasoning for Real-world Psychiatric Diagnosis},
  author={[Anonymous Authors]},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- We thank the board-certified psychiatrists who annotated the MentalDx Bench dataset
- We acknowledge the VERL framework team for providing the reinforcement learning infrastructure
- We appreciate the open-source community for their valuable tools and libraries
- The experimental and computational work in this research run on the Huawei Cloud AI Compute Service. We appreciate the stable compute supply from this platform.

---


<div align="center">

**Built with ❤️ for advancing psychiatric diagnosis with AI**

</div>
