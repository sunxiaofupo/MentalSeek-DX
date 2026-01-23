#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract diagnostic information from prediction results and evaluate against ground truth.

This script performs two steps:
1. Extract id, diagnostic_category, disorder_code and specific_disorder from prediction files
2. Evaluate the matching rate between extracted results and ground truth
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Tuple


def extract_answer_content(prediction):
    """
    Extract content within <answer></answer> tags from prediction field
    
    Args:
        prediction: String containing <answer> tags
        
    Returns:
        JSON string within answer tags, returns None if not found
    """
    # Use regular expression to extract content between <answer>...</answer>
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, prediction, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None


def parse_answer_json(answer_content):
    """
    Parse JSON string in answer content and extract required fields
    
    Args:
        answer_content: JSON string within <answer> tags
        
    Returns:
        Dictionary containing diagnostic_category, disorder_code, specific_disorder
    """
    try:
        # Parse JSON string
        answer_data = json.loads(answer_content)
        
        # Extract required fields
        result = {
            'diagnostic_category': answer_data.get('diagnostic_category', ''),
            'disorder_code': answer_data.get('disorder_code', ''),
            'specific_disorder': answer_data.get('specific_disorder', '')
        }
        
        return result
    except json.JSONDecodeError as e:
        print(f"Warning: JSON parsing error: {e}")
        print(f"Content: {answer_content[:200]}...")
        return {
            'diagnostic_category': '',
            'disorder_code': '',
            'specific_disorder': ''
        }


def extract_from_file(input_file, output_file):
    """
    Process input file, extract required information and save to output file
    
    Args:
        input_file: Input file path (JSONL format - one JSON object per line)
        output_file: Output file path (JSONL format)
        
    Returns:
        Tuple of (extracted_data_dict, error_count, missing_answer_ids)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read input file
    print(f"\n{'='*80}")
    print(f"Step 1: Extracting diagnostic information from predictions")
    print(f"{'='*80}")
    print(f"Reading file: {input_file}")
    
    # Read JSONL file (one JSON object per line)
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    print(f"Found {len(data)} records")
    
    # Process each record
    results = []
    extracted_data = {}  # Dictionary with id as key for evaluation
    error_count = 0
    missing_answer_ids = []  # Store patient ids without <answer> tags
    
    for idx, item in enumerate(data, 1):
        try:
            # Extract id
            record_id = item.get('id', '')
            
            # Extract prediction
            prediction = item.get('prediction', '')
            
            # Extract <answer> content from prediction
            answer_content = extract_answer_content(prediction)
            
            if answer_content is None:
                print(f"Warning: Record {idx} (id: {record_id}) did not find <answer> tag")
                error_count += 1
                missing_answer_ids.append(record_id)
                # Still create record, but fields are empty
                result = {
                    'id': record_id,
                    'diagnostic_category': '',
                    'disorder_code': '',
                    'specific_disorder': ''
                }
            else:
                # Parse JSON in answer
                answer_data = parse_answer_json(answer_content)
                
                # Combine results
                result = {
                    'id': record_id,
                    'diagnostic_category': answer_data['diagnostic_category'],
                    'disorder_code': answer_data['disorder_code'],
                    'specific_disorder': answer_data['specific_disorder']
                }
            
            results.append(result)
            
            # Store in dictionary for evaluation (only if id exists)
            if record_id:
                extracted_data[record_id] = result
            
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(data)} records...")
                
        except Exception as e:
            print(f"Error processing record {idx}: {e}")
            error_count += 1
            # Create empty record
            record_id = item.get('id', '')
            result = {
                'id': record_id,
                'diagnostic_category': '',
                'disorder_code': '',
                'specific_disorder': ''
            }
            results.append(result)
            if record_id:
                extracted_data[record_id] = result
    
    # Write to output file (JSONL format, one JSON object per line)
    print(f"\nWriting to output file: {output_file}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + '\n')
    
    # Save patient ids without <answer> tags
    if missing_answer_ids:
        missing_ids_file = output_path.parent / f"{output_path.stem}_missing_answer_ids.txt"
        print(f"\nSaving patient ids without <answer> tags to: {missing_ids_file}")
        with open(missing_ids_file, 'w', encoding='utf-8') as f:
            for patient_id in missing_answer_ids:
                f.write(f"{patient_id}\n")
        print(f"Saved {len(missing_answer_ids)} patient ids")
    
    print(f"\nExtraction completed!")
    print(f"Successfully processed: {len(results) - error_count} records")
    print(f"Errors/Warnings: {error_count} records")
    print(f"Output file: {output_file}")
    
    return extracted_data, error_count, missing_answer_ids


def load_jsonl(path: str, key: str) -> Dict[str, dict]:
    """
    Load JSONL file by given key and build dictionary.
    
    Args:
        path: Path to JSONL file
        key: Key to use as dictionary key (e.g., 'id')
        
    Returns:
        Dictionary with key values as keys
    """
    data: Dict[str, dict] = {}
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"File {path} line {line_num} parsing failed: {exc}") from exc

            record_id = str(record.get(key, "")).strip()
            if not record_id:
                print(f"Warning: File {path} line {line_num} missing {key}, skipping")
                continue
            data[record_id] = record
    return data


def evaluate(predictions: Dict[str, dict], ground_truth: Dict[str, dict]) -> Tuple[int, int, int, int, int]:
    """
    Evaluate matching rate between predictions and ground truth.
    
    Args:
        predictions: Dictionary of predictions {id: record}
        ground_truth: Dictionary of ground truth {id: record}
        
    Returns:
        Tuple of (GT total, common ID count, diagnostic match count, code match count, full match count)
    """
    common_ids = set(predictions.keys()) & set(ground_truth.keys())
    gt_total = len(ground_truth)

    diag_match = 0
    code_match = 0
    full_match = 0

    for pid in common_ids:
        pred = predictions[pid]
        gt = ground_truth[pid]

        pred_diag = str(pred.get("diagnostic_category", "")).strip()
        pred_code = str(pred.get("disorder_code", "")).strip()

        gt_diag = str(gt.get("diagnostic_category", "")).strip()
        gt_code = str(gt.get("disorder_code", "")).strip()

        diag_correct = pred_diag == gt_diag
        code_correct = pred_code == gt_code

        if diag_correct:
            diag_match += 1
        if code_correct:
            code_match += 1
        if diag_correct and code_correct:
            full_match += 1

    return gt_total, len(common_ids), diag_match, code_match, full_match


def main():
    """Main function: Extract and evaluate"""
    # File paths configuration - resolve based on script location to get project root
    script_dir = Path(__file__).parent.absolute()  # eval/src/
    eval_dir = script_dir.parent  # eval/
    project_root = eval_dir.parent  # MentalSeek-Dx/
    
    # File paths relative to project root (MentalSeek-Dx/)
    input_file = project_root / 'eval' / 'prediction_output' / 'MentalSeek-Dx-7B.jsonl'
    output_file = project_root / 'eval' / 'extracted_result' / 'MentalSeek-Dx-7B.jsonl'
    # Ground truth: MentalDx-Bench/gt.jsonl
    gt_file = project_root / 'MentalDx-Bench' / 'Ground_Truth.jsonl'
    
    # Convert to string for compatibility
    input_file = str(input_file)
    output_file = str(output_file)
    gt_file = str(gt_file)
    
    try:
        # Step 1: Extract diagnostic information from predictions
        extracted_data, error_count, missing_ids = extract_from_file(input_file, output_file)
        
        if not extracted_data:
            print("Error: No data extracted, cannot proceed to evaluation")
            return 1
        
        # Step 2: Load ground truth and evaluate
        print(f"\n{'='*80}")
        print(f"Step 2: Evaluating extracted results against ground truth")
        print(f"{'='*80}")
        print(f"Loading ground truth file: {gt_file}")
        
        ground_truth = load_jsonl(gt_file, "id")
        print(f"Ground truth file has {len(ground_truth)} records")
        
        # Evaluate
        gt_total, common_total, diag_match, code_match, full_match = evaluate(
            extracted_data, ground_truth
        )

        if gt_total == 0:
            print("Error: Ground truth is empty, cannot calculate accuracy.")
            return 1

        diag_accuracy = diag_match / gt_total
        code_accuracy = code_match / gt_total
        full_accuracy = full_match / gt_total

        # Print evaluation results
        print(f"\n{'='*80}")
        print(f"Evaluation Results:")
        print(f"{'='*80}")
        print(f"Ground truth total: {gt_total}")
        print(f"Common ID count: {common_total}")
        print(f"Extracted records: {len(extracted_data)}")
        print(f"\nAccuracy Metrics:")
        print(f"  Diagnostic category matches: {diag_match} / {gt_total} = {diag_accuracy:.4%}")
        print(f"  Disorder code matches: {code_match} / {gt_total} = {code_accuracy:.4%}")
        print(f"  Full matches (both category and code): {full_match} / {gt_total} = {full_accuracy:.4%}")
        
        if missing_ids:
            print(f"\nWarning: {len(missing_ids)} records had missing <answer> tags")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
