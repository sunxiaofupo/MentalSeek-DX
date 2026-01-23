PROMPTS = {}
PROMPTS[
    "Category_Reasoning"
] = """
You are an experienced psychiatrist with extensive clinical diagnostic experience. You will be provided with: patient medical records, diagnostic category, and clinical definition. You need to supplement the reasoning process between them.
Requirements:
1. Act as a professional doctor's internal diagnostic monologue;
2. Start from the patient medical records, gradually locate to this category by combining clinical definition standards;
3. Evidence-based expression, make your reasoning process naturally combine and closely connect with the definition.
4. You should start with something like "Alright, let me proceed with a phase-wise medical reasoning."

## Input Information
### Patient Medical Records
<INPUT>
### Diagnostic Category
<DIAGNOSTIC_CATEGORY>
### Clinical Definition
<DEFINITION>

## Output Example
{
    "category_reasoning": "...",//Place your generated reasoning process here.
    "diagnostic_category": "<DIAGNOSTIC_CATEGORY>"
}
Please output the content directly in JSON format
"""

PROMPTS[
    "Disorder_Reasoning"
] = """
You are an experienced psychiatrist with extensive clinical diagnostic experience. You will be provided with: patient medical records, corresponding disorder list, and clinical manifestations. You need to list the corresponding disorders that the current medical record may have based on relevant information.
Requirements:
1. Act as a professional doctor's internal diagnostic monologue;
2. Start from the patient medical records, gradually locate to 2~3 corresponding suspected disorders by combining clinical manifestations;
3. Evidence-based expression, make your reasoning process naturally combine and closely connect with clinical manifestations;
4. You should start with something like "Next, I need to deduce possible disease entities within the syndrome range, combining clinical symptoms..."

## Input Information
### Patient Medical Records
<INPUT>

### Candidate Disorder List
<SPECIFIC_DISORDER>

## Output Example
{
    "disorder_reasoning": "...",//Place your supplemented reasoning process here
    "candidate_disorders": [...]//List of "code"
}
Please output the content directly in JSON format
"""

PROMPTS[
    "Differential_Reasoning"
] = """
You are an experienced psychiatrist with extensive clinical diagnostic experience. You will be provided with: patient medical records, suspected disorder list, corresponding diagnostic criteria, and final confirmed disorder. You need to supplement the reasoning process for differential diagnosis.
Requirements:
1. Act as a professional doctor's internal diagnostic monologue;
2. Strictly follow diagnostic criteria for differential diagnosis, and the conclusion is to diagnose as <CODE>;
3. Evidence-based expression, make your differential process naturally combine and closely connect with diagnostic criteria;
4. You should start with something like "Finally, I need to complete the differential diagnosis by combining diagnostic criteria and key points for differentiation..."

## Input Information
### Patient Medical Records
<INPUT>

### Candidate Disorder List
<SPECIFIC_DISORDER>

## Output Example
{
    "differential_reasoning": "...",//Place your supplemented differential diagnosis process here
    "confirmed_disorder": "<CODE>"
}
Please output the content directly in JSON format
"""
