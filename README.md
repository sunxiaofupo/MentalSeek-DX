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