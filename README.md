# MentalSeek-Dx Project Documentation

This document describes the purpose and usage of each file and directory in the MentalSeek-Dx project.

## Project Structure

```
MentalSeek-Dx/
├── eval/                          # Evaluation related files
│   ├── data/                      # Evaluation data
│   │   ├── Bench-test-result/     # Test result files
│   │   └── Result-extracted/      # Extracted result files
│   ├── src/                       # Evaluation scripts
│   │   ├── extract_and_evaluate.py
│   │   └── statistics_category_accuracy.py
│   └── category_accuracy_statistics.txt  # Statistics results
├── GRPO_CRL/                      # Reinforcement learning related
│   └── reward_function/           # Reward function
│       └── Progressive_Process-based_reward.py
├── MentalDx-Bench/                # Benchmark test data
│   ├── gt.jsonl                   # Ground Truth (standard answers)
│   └── MentalDx-Bench.jsonl       # Benchmark dataset
├── Prompt/                        # Prompt templates
│   ├── diagnosis_thinking.txt
│   └── translation_and_anonymity.txt
└── Structured_Psychiatric_Knowledge_Base/  # Structured psychiatric knowledge base
    ├── Criteria.jsonl             # Diagnostic criteria
    └── Diagnostic_category.jsonl  # Diagnostic categories
```

---

## Detailed Description of Directories and Files

### 1. eval/ - Evaluation Module

#### 1.1 eval/src/extract_and_evaluate.py

**Function**: Extract diagnostic information from prediction results and evaluate accuracy

**Main Features**:
1. Extract diagnostic information from `<answer>` tags in prediction files
2. Extract fields: `id`, `diagnostic_category`, `disorder_code`, `specific_disorder`
3. Match and evaluate against Ground Truth

**Usage**:
```bash
cd MentalSeek-Dx/eval/src
python extract_and_evaluate.py
```

**Input Files**:
- `eval/data/Bench-test-result/MentalSeek-Dx-7B.jsonl` - Model prediction results (JSONL format)

**Output Files**:
- `eval/data/Result-extracted/MentalSeek-Dx-7B.jsonl` - Extracted diagnostic information (JSONL format)
- `eval/data/Result-extracted/MentalSeek-Dx-7B_missing_answer_ids.txt` - Record IDs missing `<answer>` tags

**Evaluation Metrics**:
- Diagnostic category matching rate (Diagnostic category accuracy)
- Disorder code matching rate (Disorder code accuracy)
- Full match rate (Full match accuracy - both category and code match)

**Output Example**:
```
================================================================================
Evaluation Results:
================================================================================
Ground truth total: 712
Common ID count: 712
Extracted records: 712

Accuracy Metrics:
  Diagnostic category matches: 588 / 712 = 82.58%
  Disorder code matches: 481 / 712 = 67.56%
  Full matches (both category and code): 456 / 712 = 64.04%
```

---

#### 1.2 eval/src/statistics_category_accuracy.py

**Function**: Calculate accuracy statistics for different diagnostic categories

**Main Features**:
1. Traverse all result files in the `Result-extracted/` directory
2. Calculate accuracy for each diagnostic category (diagnostic_category)
3. Generate detailed statistics report

**Usage**:
```bash
cd MentalSeek-Dx
python eval/src/statistics_category_accuracy.py
```

**Input Files**:
- `eval/data/Result-extracted/*.jsonl` - All extracted result files
- `MentalDx-Bench/gt.jsonl` - Ground Truth

**Output Files**:
- `eval/category_accuracy_statistics.txt` - Statistics report

**Output Content**:
- Overall statistics (disorder_code and diagnostic_category matching accuracy)
- Detailed statistics for each diagnostic category
- Match counts and accuracy rates for each category

**Output Example**:
```
====================================================================================================
Overall Statistics (disorder_code matching accuracy):
====================================================================================================

Filename                                           Common ID    Matches      Accuracy    
----------------------------------------------------------------------------------------------------
MentalSeek-Dx-14B.jsonl                            712          499          70.08%        
MentalSeek-Dx-7B.jsonl                             712          481          67.56%        
```

---

#### 1.3 eval/data/ - Evaluation Data Directory

**Bench-test-result/** - Test result files
- Contains raw model prediction results (JSONL format)
- Each line contains `id` and `prediction` fields
- `prediction` field contains `<think>...</think>` and `<answer>...</answer>` tags

**Result-extracted/** - Extracted result files
- Diagnostic information extracted from prediction results
- Each line contains: `id`, `diagnostic_category`, `disorder_code`, `specific_disorder`

---

### 2. MentalDx-Bench/ - Benchmark Test Data

#### 2.1 gt.jsonl

**Function**: Ground Truth file (standard answers)

**Format**: JSONL format, one JSON object per line
```json
{"id": "1000769", "disorder_code": "6A20", "diagnostic_category": "Schizophrenia or other primary psychotic disorders", "English_disease_name": "Schizophrenia"}
```

**Field Description**:
- `id`: Patient ID
- `disorder_code`: Disorder code (ICD-11)
- `diagnostic_category`: Diagnostic category
- `English_disease_name`: English disease name

**Usage**:
- Used as reference standard for evaluating model prediction accuracy
- Used to calculate matching rates and accuracy

---

#### 2.2 MentalDx-Bench.jsonl

**Function**: Benchmark dataset

**Format**: JSONL format, containing test cases

**Usage**:
- Used for model testing and evaluation
- Contains input data for test cases

---

### 3. Prompt/ - Prompt Templates

#### 3.1 diagnosis_thinking.txt

**Function**: Diagnostic thinking chain prompt template

**Usage**:
- Used to guide the model in diagnostic reasoning
- Contains Chain of Thought prompt format

---

#### 3.2 translation_and_anonymity.txt

**Function**: Translation and anonymization prompt template

**Usage**:
- Used for data translation and anonymization processing
- Guides the model in medical text translation

---

### 4. Structured_Psychiatric_Knowledge_Base/ - Structured Knowledge Base

#### 4.1 Criteria.jsonl

**Function**: Diagnostic criteria knowledge base

**Format**: JSONL format, one diagnostic criterion per line
```json
{
  "code": "6A70",
  "diagnostic_category": "...",
  "specific_disorder": "...",
  "chinese_name": "...",
  "definition": "...",
  "clinical_manifestations": "...",
  "symptom": "...",
  "diagnostic_criteria": "..."
}
```

**Field Description**:
- `code`: Disorder code
- `diagnostic_category`: Diagnostic category
- `specific_disorder`: Specific disorder
- `chinese_name`: Chinese name
- `definition`: Definition
- `clinical_manifestations`: Clinical manifestations
- `symptom`: Symptoms
- `diagnostic_criteria`: Diagnostic criteria

**Usage**:
- Used as diagnostic reference knowledge base
- Used for model training and inference

---

#### 4.2 Diagnostic_category.jsonl

**Function**: Diagnostic category definition file

**Format**: JSONL format, one diagnostic category per line
```json
{
  "diagnostic_category": "Neurodevelopmental disorders",
  "code_list": ["6A00", "6A01", "6A02", "6A03", "6A04", "6A05", "6A06"],
  "Definition": "Neurodevelopmental disorders are a group of mental or behavioral disorders that appear during the stages of individual development..."
}
```

**Field Description**:
- `diagnostic_category`: Diagnostic category (English)
- `code_list`: List of disorder codes under this category
- `Definition`: Category definition (Chinese)

**Usage**:
- Defines the scope and meaning of each diagnostic category
- Used for classification and statistics

---

### 5. GRPO_CRL/ - Reinforcement Learning Module

#### 5.1 GRPO_CRL/reward_function/Progressive_Process-based_reward.py

**Function**: Progressive process-based reward function

**Main Features**:
- Implements process-based reward calculation
- Supports Curriculum Learning
- Calculates diagnostic accuracy rewards

**Usage**:
- Used for reinforcement learning training
- Evaluates the quality of model reasoning process

---

## Usage Workflow Examples

### Complete Evaluation Workflow

1. **Prepare Prediction Results**
   - Save model prediction results to the `eval/data/Bench-test-result/` directory

2. **Extract Diagnostic Information**
   ```bash
   cd MentalSeek-Dx/eval/src
   python extract_and_evaluate.py
   ```
   - Extract diagnostic information from prediction results
   - Generate extracted result files

3. **Calculate Accuracy Statistics**
   ```bash
   cd MentalSeek-Dx
   python eval/src/statistics_category_accuracy.py
   ```
   - Calculate overall and category-specific accuracy
   - Generate statistics report

---

## Notes

1. **File Format**:
   - All `.jsonl` files are in JSON Lines format (one JSON object per line)
   - Not standard JSON array format

2. **Path Configuration**:
   - Scripts use relative paths, recommended to run from project root directory
   - Paths are automatically resolved based on script location

3. **Dependencies**:
   - Python 3.7+
   - Required dependency packages (e.g., `tqdm`, etc.)

4. **Data Format**:
   - Prediction results must contain `<answer>` tags
   - Content within tags should be valid JSON format
