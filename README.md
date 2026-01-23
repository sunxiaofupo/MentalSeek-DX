# MentalSeek-Dx: Towards Progressive Hypothetico-Deductive Reasoning for Real-world Psychiatric Diagnosis

## Introduction

MentalSeek-Dx is a medical-specialized large language model (LLM) designed to address real-world psychiatric diagnosis, with a focus on aligning AI reasoning with clinical practices. Mental health disorders are a growing global public health concern. While LLMs are promising for psychiatric assessment, existing benchmarks often lack real-world complexity and detailed diagnostic supervision, which limits their clinical applicability.

To address this, we introduce **MentalDx Bench** — the first benchmark targeting disorder-level psychiatric diagnosis in real clinical settings. The benchmark consists of 712 de-identified electronic health records, each annotated by board-certified psychiatrists using ICD-11 guidelines, covering 76 disorders across 16 diagnostic categories.

Evaluation of 18 LLMs identified a "paradigm misalignment": while many models perform well at broad diagnostic categorization, they struggle with fine-grained, disorder-level diagnosis, revealing a gap between pattern recognition and rigorous clinical reasoning.

MentalSeek-Dx bridges this gap by internalizing clinical hypothetico-deductive reasoning. It is trained using supervised trajectory construction and curriculum-based reinforcement learning to more faithfully simulate a psychiatrist’s diagnostic process.

**Key Highlights:**
- State-of-the-art (SOTA) performance on the MentalDx Bench with only 14B parameters
- Deep integration of clinical reasoning, moving beyond pattern-based diagnosis
- Establishes a practical, trustworthy foundation for psychiatric diagnosis with LLMs

For more details, usage instructions, and evaluation steps, see below.

## Project Structure

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
│       └── statistics_category_accuracy.py       # Category accuracy statistics script
│
├── Models/                                      # Model configurations and documentation
│   ├── MentalSeek-Dx-14B/
│   │   └── models.md                            # Documentation for 14B model
│   └── MentalSeek-Dx-7B/
│       └── models.md                            # Documentation for 7B model
│
├── eval_results/                                # Evaluation results directory
│   ├── extracted_result/                        # Extracted evaluation results
│   │   ├── MentalSeek-Dx-14B.jsonl              # 14B model evaluation results
│   │   └── MentalSeek-Dx-7B.jsonl               # 7B model evaluation results
│   ├── prediction_output/                       # Model prediction outputs
│   │   ├── MentalSeek-Dx-14B.jsonl              # 14B model predictions
│   │   └── MentalSeek-Dx-7B.jsonl               # 7B model predictions
│   └── result_statistics.txt                    # Aggregated result statistics
│
└── verl/                                        # VERL framework (third-party reinforcement learning toolkit)
    ├
    ├── ...
    ├── recipe/                                  # Recipe/configuration
    │   └── MentalSeek-Dx/                       # MentalSeek-Dx specific configuration
    ├── requirements.txt                         # VERL dependencies
    ├── setup.py                                 # Installation script
    └── README.md                                # VERL framework documentation
```

## Component Overview

### Core Modules

- **`run_MentalSeek-Dx.py`**: Main entry script for running the MentalSeek-Dx model.
- **`Reasoning_Trajectory_Building/`**: Contains the essential logic for constructing reasoning trajectories and related prompt templates.
- **`Structured_Psychiatric_Knowledge_Base/`**: Stores structured psychiatric knowledge in JSONL format.

### Evaluation Modules

- **`MentalDx-Bench/`**: Includes the benchmark dataset and evaluation utilities.
  - `Ground_Truth.jsonl`: Gold-standard diagnostic labels.
  - `MentalDx-Bench.jsonl`: Full benchmark dataset (712 records).
  - `src/`: Source code for evaluation scripts.

### Results Output

- **`eval_results/`**: Stores model evaluation results.
  - `extracted_result/`: Extracted evaluation results (in JSONL format).
  - `prediction_output/`: Model output predictions.
  - `result_statistics.txt`: Aggregated statistical summaries.

### Model Configurations

- **`Models/`**: Contains configuration files and documentation for models of varying sizes.
  - `MentalSeek-Dx-14B/`: Configuration and documentation for the 14B-parameter model.
  - `MentalSeek-Dx-7B/`: Configuration and documentation for the 7B-parameter model.

### Third-party Framework

- **`verl/`**: VERL (Versatile Reinforcement Learning) framework.
  - Provides the foundational reinforcement learning framework for training.
  - Includes trainers, worker nodes, utility functions, and comprehensive implementations.
  - Supports multiple training algorithms (e.g., PPO, GRPO, GPG, etc.).


## Environment Setup

Follow the steps below to set up your environment:

```bash
# 1. Create a Python 3.11 environment
conda create -n mentalseek-dx python==3.11

# 2. Configure VERL
# (For detailed installation and setup instructions, please refer to the official VERL documentation.)
# VERL repository: https://github.com/volcengine/verl

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Model Weights

| Model Name         | Weight Availability                                                         |
|--------------------|-----------------------------------------------------------------------------|
| MentalSeek-Dx-7B   | Our model will be made publicly available upon the acceptance of the paper. |
| MentalSeek-Dx-14B  | Our model will be made publicly available upon the acceptance of the paper. |

---

### Benchmark Data

- **Input File**: `MentalSeek-Dx/MentalDx-Bench/MentalDx-Bench.jsonl`
- **Ground Truth Labels**: `MentalSeek-Dx/MentalDx-Bench/Ground_Truth.jsonl`

---

## Inference Example

To perform inference with MentalSeek-Dx-7B, use the following code:

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "path/to/MentalSeek-Dx-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """
You are a psychiatric diagnosis expert. I will provide you with the patient's medical records. Please analyze the medical record information I provide and give the corresponding diagnosis results, including the category, specific disorder, and corresponding ICD-11 code (must be precise to four digits).

## Diagnostic Category Reference: ["Neurodevelopmental disorders","Schizophrenia or other primary psychotic disorders","Catatonia","Mood disorders","Anxiety or fear-related disorders","Obsessive-compulsive or related disorders","Disorders specifically associated with stress","Dissociative disorders","Feeding or eating disorders","Elimination disorders","Disorders of bodily distress or bodily experience","Disorders due to substance use or addictive behaviours","Impulse control disorders","Disruptive behaviour or dissocial disorders","Personality disorders and related traits","Paraphilic disorders","Neurocognitive disorders","Mental or behavioural disorders associated with pregnancy, childbirth or the puerperium","Factitious disorders"]

## Specific Disorders and Corresponding ICD-11 Code Reference:
["6A20 Schizophrenia","6B00 Generalised anxiety disorder","6A72 Dysthymic disorder","6E20 Mental or behavioural disorders associated with pregnancy, childbirth or the puerperium, without psychotic symptoms","6A61 Bipolar type II disorder","6A00 Disorders of intellectual development","6D70 Delirium","6E21 Mental or behavioural disorders associated with pregnancy, childbirth or the puerperium, with psychotic symptoms","6B20 Obsessive-compulsive disorder","6A40 Catatonia associated with another mental disorder","6D72 Amnestic disorder","6A62 Cyclothymic disorder","6A23 Acute and transient psychotic disorder","6B43 Adjustment disorder","6A21 Schizoaffective disorder","6A70 Single episode depressive disorder","6A73 Mixed depressive and anxiety disorder","6A71 Recurrent depressive disorder","6A24 Delusional disorder","6A60 Bipolar type I disorder","6C20 Bodily distress disorder","6A22 Schizotypal disorder","6A41 Catatonia induced by substances or medications","6D71 Mild neurocognitive disorder","6D85 Dementia due to diseases classified elsewhere","6C4G Disorders due to use of unknown or unspecified psychoactive substances","6C4E Disorders due to use of other specified psychoactive substances, including medications","6C46 Disorders due to use of stimulants including amphetamines, methamphetamine or methcathinone","6B04 Social anxiety disorder","6C40 Disorders due to use of alcohol","6D10 Personality disorder","6A02 Autism spectrum disorder","6B02 Agoraphobia","6B60 Dissociative neurological symptom disorder","6C91 Conduct-dissocial disorder","6C90 Oppositional defiant disorder","6B83 Avoidant-restrictive food intake disorder","6B40 Post traumatic stress disorder","6C4F Disorders due to use of multiple specified psychoactive substances, including medications","6D84 Dementia due to psychoactive substances including medications","6B63 Possession trance disorder","6B03 Specific phobia","6B41 Complex post traumatic stress disorder","6A03 Developmental learning disorder","6C51 Gaming disorder","6D86 Behavioural or psychological disturbances in dementia","6A06 Stereotyped movement disorder","6D83 Frontotemporal dementia","6B80 Anorexia Nervosa","6B23 Hypochondriasis","6C49 Disorders due to use of hallucinogens","6B81 Bulimia Nervosa","6A05 Attention deficit hyperactivity disorder","6B62 Trance disorder","6B84 Pica","6B21 Body dysmorphic disorder","6B05 Separation anxiety disorder","6B64 Dissociative identity disorder","6B61 Dissociative amnesia","6A04 Developmental motor coordination disorder","6C4D Disorders due to use of dissociative drugs including ketamine and phencyclidine [PCP]","6B66 Depersonalization-derealization disorder","6B25 Body-focused repetitive behaviour disorders","6B42 Prolonged grief disorder","6B82 Binge eating disorder","6C44 Disorders due to use of sedatives, hypnotics or anxiolytics","6B22 Olfactory reference disorder","6C01 Encopresis","6D81 Dementia due to cerebrovascular disease","6B06 Selective mutism","6C50 Gambling disorder","6B24 Hoarding disorder","6C41 Disorders due to use of cannabis","6B85 Rumination-regurgitation disorder","6C47 Disorders due to use of synthetic cathinones"]

## Output Format:
You need to wrap your diagnostic process with <think></think> placeholders and wrap the final diagnosis result with <answer></answer> placeholders. The diagnosis result should be output in JSON format, where:
 - "diagnostic_category": Select the most likely category from the diagnostic categories listed above;
-  "candidate_disorders": A list of candidate disorders you consider as suspected, only include the codes of candidate disorders;
-  "disorder_code": The final confirmed disorder code;
- "specific_disorder": The final confirmed disorder name, according to the official ICD-11 name;

## Output Example:
-------------
<think>...</think>
<answer>
{
    "diagnostic_category": "Disorders specifically associated with stress",
    "candidate_disorders": ["6B43", "6B40"],
    "disorder_code": "6B43",
    "specific_disorder": "Adjustment disorder"
}
</answer>

## Test Medical Record: {MEDICAL_RECORD}
"""

messages = [
    {"role": "system", "content": "You are MentalSeek-Dx,a psychiatric diagnosis expert"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

---

## Model Training Process

- **Step 1: Cold Start Data Preparation**  
  Path: `MentalSeek-Dx/Reasoning_Trajectory_Building`

- **Step 2: Progressive HDR Training**  
  Path: `MentalSeek-Dx/verl/recipe/MentalSeek-Dx/train.sh`

---

## Evaluation

- Run the evaluation script:  
  `MentalSeek-Dx/MentalDx-Bench/src/extract_and_evaluate.py`
