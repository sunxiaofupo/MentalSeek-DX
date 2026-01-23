#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistics script: Calculate matching accuracy between jsonl files in result-extracted folder and gt file
Including disorder_code matching accuracy and diagnostic accuracy for each category in diagnostic_category
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter


def load_jsonl(file_path):
    """Load JSONL file and return a dictionary with id as key"""
    data = {}
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                data[record['id']] = record
                # Save first 3 records as examples
                if i <= 3:
                    examples.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parsing failed at line {i} in file {file_path}: {e}")
    return data, examples


def calculate_code_accuracy(result_data, gt_data):
    """
    Calculate disorder_code matching accuracy
    Accuracy = number of matches / total number in baseline (gt)
    
    Args:
        result_data: Dictionary of result file {id: record}
        gt_data: Dictionary of gt file {id: record}
    
    Returns:
        dict: Dictionary containing matching statistics
    """
    # Find common ids
    common_ids = set(result_data.keys()) & set(gt_data.keys())
    total_common = len(common_ids)
    gt_total = len(gt_data)
    
    if gt_total == 0:
        print("Warning: GT data is empty, cannot calculate accuracy")
        return {
            'total_common': 0,
            'gt_total': 0,
            'code_match': 0,
            'code_match_rate': 0.0
        }
    
    code_match_count = 0
    
    for patient_id in common_ids:
        result_code = result_data[patient_id].get('disorder_code', '')
        gt_code = gt_data[patient_id].get('disorder_code', '')
        if result_code == gt_code:
            code_match_count += 1
    
    return {
        'total_common': total_common,
        'gt_total': gt_total,
        'code_match': code_match_count,
        'code_match_rate': (code_match_count / gt_total * 100) if gt_total > 0 else 0.0
    }


def calculate_category_accuracy(result_data, gt_data):
    """
    Calculate diagnostic_category matching accuracy
    Accuracy = number of matches / total number in baseline (gt)
    
    Args:
        result_data: Dictionary of result file {id: record}
        gt_data: Dictionary of gt file {id: record}
    
    Returns:
        dict: Dictionary containing matching statistics
    """
    # Find common ids
    common_ids = set(result_data.keys()) & set(gt_data.keys())
    total_common = len(common_ids)
    gt_total = len(gt_data)
    
    if gt_total == 0:
        print("Warning: GT data is empty, cannot calculate accuracy")
        return {
            'total_common': 0,
            'gt_total': 0,
            'category_match': 0,
            'category_match_rate': 0.0
        }
    
    category_match_count = 0
    
    for patient_id in common_ids:
        result_category = result_data[patient_id].get('diagnostic_category', '')
        gt_category = gt_data[patient_id].get('diagnostic_category', '')
        if result_category == gt_category:
            category_match_count += 1
    
    return {
        'total_common': total_common,
        'gt_total': gt_total,
        'category_match': category_match_count,
        'category_match_rate': (category_match_count / gt_total * 100) if gt_total > 0 else 0.0
    }


def calculate_both_accuracy(result_data, gt_data):
    """
    Calculate accuracy when both disorder_code and diagnostic_category match
    Accuracy = number of matches / total number in baseline (gt)
    
    Args:
        result_data: Dictionary of result file {id: record}
        gt_data: Dictionary of gt file {id: record}
    
    Returns:
        dict: Dictionary containing matching statistics
    """
    # Find common ids
    common_ids = set(result_data.keys()) & set(gt_data.keys())
    total_common = len(common_ids)
    gt_total = len(gt_data)
    
    if gt_total == 0:
        print("Warning: GT data is empty, cannot calculate accuracy")
        return {
            'total_common': 0,
            'gt_total': 0,
            'both_match': 0,
            'both_match_rate': 0.0
        }
    
    both_match_count = 0
    
    for patient_id in common_ids:
        result_code = result_data[patient_id].get('disorder_code', '')
        gt_code = gt_data[patient_id].get('disorder_code', '')
        result_category = result_data[patient_id].get('diagnostic_category', '')
        gt_category = gt_data[patient_id].get('diagnostic_category', '')
        
        # Match both disorder_code and diagnostic_category
        if result_code == gt_code and result_category == gt_category:
            both_match_count += 1
    
    return {
        'total_common': total_common,
        'gt_total': gt_total,
        'both_match': both_match_count,
        'both_match_rate': (both_match_count / gt_total * 100) if gt_total > 0 else 0.0
    }


def get_category_abbreviation():
    """
    Return mapping dictionary from category names to abbreviations
    """
    return {
        "Anxiety or fear-related disorders": "Anxiety",
        "Catatonia": "Catatonia",
        "Disorders due to substance use or addictive behaviours": "Substance",
        "Disorders of bodily distress or bodily experience": "Bodily",
        "Disorders specifically associated with stress": "Stress",
        "Disruptive behaviour or dissocial disorders": "Disruptive",
        "Dissociative disorders": "Dissociative",
        "Elimination disorders": "Elimination",
        "Feeding or eating disorders": "Eating",
        "Mental or behavioural disorders associated with pregnancy, childbirth or the puerperium": "Pregnancy",
        "Mood disorders": "Mood",
        "Neurocognitive disorders": "Neurocognitive",
        "Neurodevelopmental disorders": "Neurodevelopmental",
        "Obsessive-compulsive or related disorders": "OCD",
        "Personality disorders and related traits": "Personality",
        "Schizophrenia or other primary psychotic disorders": "Schizophrenia"
    }


def calculate_category_accuracy_by_category(result_data, gt_data):
    """
    Calculate diagnostic accuracy for each diagnostic_category
    Accuracy = number of matches / total number of this category in baseline (gt)
    
    Args:
        result_data: Dictionary of result file {id: record}
        gt_data: Dictionary of gt file {id: record}
    
    Returns:
        dict: {category: {'total': count, 'match': count, 'gt_total': count, 'rate': rate}}
    """
    # Find common ids
    common_ids = set(result_data.keys()) & set(gt_data.keys())
    
    # Count total number of each category in gt (as denominator)
    gt_category_total = defaultdict(int)
    for patient_id in gt_data.keys():
        gt_category = gt_data[patient_id].get('diagnostic_category', 'Unknown Category')
        gt_category_total[gt_category] += 1
    
    # Count matches by category
    category_match_count = defaultdict(int)
    
    for patient_id in common_ids:
        result_record = result_data[patient_id]
        gt_record = gt_data[patient_id]
        
        # Get category
        gt_category = gt_record.get('diagnostic_category', 'Unknown Category')
        
        # Check if disorder_code matches
        result_code = result_record.get('disorder_code', '')
        gt_code = gt_record.get('disorder_code', '')
        code_match = (result_code == gt_code)
        
        # Count matches for this category
        if code_match:
            category_match_count[gt_category] += 1
    
    # Calculate accuracy (denominator uses total number of this category in gt)
    category_accuracy = {}
    for category in gt_category_total.keys():
        gt_total = gt_category_total[category]
        match = category_match_count.get(category, 0)
        rate = (match / gt_total * 100) if gt_total > 0 else 0.0
        category_accuracy[category] = {
            'match': match,
            'gt_total': gt_total,  # Total number of this category in gt (as denominator)
            'rate': rate
        }
    
    return category_accuracy


def main():
    # File paths
    # Note: These paths are relative to the project root directory (MentalSeek-Dx/)
    # Script should be run from the MentalSeek-Dx/ directory
    result_dir = "./MentalSeek-Dx/eval/data/Result-extracted"
    gt_file = "./MentalSeek-Dx/MentalDx-Bench/Ground_Truth.jsonl"
    output_file = "./MentalSeek-Dx/eval/category_accuracy_statistics.txt"
    
    print("=" * 100)
    print("Step 1: Traverse all subfiles in result-extracted folder")
    print("=" * 100)
    
    # Get all jsonl files (excluding gt.jsonl)
    result_dir_path = Path(result_dir)
    jsonl_files = [f for f in result_dir_path.glob("*.jsonl") if f.name != "gt.jsonl"]
    jsonl_files.sort()
    
    if not jsonl_files:
        print(f"No jsonl files found in {result_dir}")
        return
    
    print(f"\nFound {len(jsonl_files)} result files")
    
    # Display example content for each file
    for jsonl_file in jsonl_files:
        print(f"\nFile: {jsonl_file.name}")
        result_data, examples = load_jsonl(str(jsonl_file))
        print(f"  Total {len(result_data)} records")
        print(f"  Example content (first 3 records):")
        for i, example in enumerate(examples, 1):
            print(f"    Example {i}:")
            print(f"      {json.dumps(example, ensure_ascii=False, indent=6)}")
    
    print("\n" + "=" * 100)
    print("Step 2: Traverse gt.jsonl file")
    print("=" * 100)
    
    # Load gt data
    print("\nLoading gt file...")
    gt_data, gt_examples = load_jsonl(gt_file)
    print(f"gt file has {len(gt_data)} records")
    print(f"\nExample content (first 3 records):")
    for i, example in enumerate(gt_examples, 1):
        print(f"  Example {i}:")
        print(f"    {json.dumps(example, ensure_ascii=False, indent=4)}")
    
    print("\n" + "=" * 100)
    print("Step 3: Calculate matching accuracy")
    print("=" * 100)
    
    # Statistics results
    all_results = []
    
    # Process each file
    for jsonl_file in jsonl_files:
        print(f"\nProcessing file: {jsonl_file.name}")
        
        # Load result file
        result_data, _ = load_jsonl(str(jsonl_file))
        
        # Calculate disorder_code matching accuracy
        code_stats = calculate_code_accuracy(result_data, gt_data)
        
        # Calculate diagnostic_category matching accuracy
        category_stats = calculate_category_accuracy(result_data, gt_data)
        
        # Calculate accuracy when both disorder_code and diagnostic_category match
        both_stats = calculate_both_accuracy(result_data, gt_data)
        
        # Calculate diagnostic accuracy for each category
        category_accuracy = calculate_category_accuracy_by_category(result_data, gt_data)
        
        # Save results
        result_info = {
            'filename': jsonl_file.name,
            'result_count': len(result_data),
            'code_stats': code_stats,
            'category_stats': category_stats,
            'both_stats': both_stats,
            'category_accuracy': category_accuracy
        }
        all_results.append(result_info)
        
        # Print statistics
        print(f"  Common ID count: {code_stats['total_common']}")
        print(f"  disorder_code match: {code_stats['code_match']} ({code_stats['code_match_rate']:.2f}%)")
        print(f"  diagnostic_category match: {category_stats['category_match']} ({category_stats['category_match_rate']:.2f}%)")
        print(f"  Both match (disorder_code+diagnostic_category): {both_stats['both_match']} ({both_stats['both_match_rate']:.2f}%)")
        print(f"  Category count: {len(category_accuracy)}")
    
    # Save statistics to file
    print(f"\n" + "=" * 100)
    print(f"Step 4: Save statistics to file")
    print("=" * 100)
    print(f"Saving statistics to: {output_file}")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("Statistics: Matching accuracy between jsonl files in result-extracted folder and gt file\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"GT file: {gt_file}\n")
        f.write(f"GT file record count: {len(gt_data)}\n")
        f.write(f"Number of statistics files: {len(jsonl_files)}\n\n")
        
        # Overall statistics
        f.write("=" * 100 + "\n")
        f.write("Overall Statistics (disorder_code matching accuracy):\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"{'Filename':<50} {'Common ID':<12} {'Matches':<12} {'Accuracy':<12}\n")
        f.write("-" * 100 + "\n")
        
        for result in all_results:
            code_stats = result['code_stats']
            f.write(f"{result['filename']:<50} ")
            f.write(f"{code_stats['total_common']:<12} ")
            f.write(f"{code_stats['code_match']:<12} ")
            f.write(f"{code_stats['code_match_rate']:.2f}%{'':<8}\n")
        
        # diagnostic_category matching accuracy statistics
        f.write("\n" + "=" * 100 + "\n")
        f.write("Overall Statistics (diagnostic_category matching accuracy):\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"{'Filename':<50} {'Common ID':<12} {'Matches':<12} {'Accuracy':<12}\n")
        f.write("-" * 100 + "\n")
        
        for result in all_results:
            category_stats = result['category_stats']
            f.write(f"{result['filename']:<50} ")
            f.write(f"{category_stats['total_common']:<12} ")
            f.write(f"{category_stats['category_match']:<12} ")
            f.write(f"{category_stats['category_match_rate']:.2f}%{'':<8}\n")
        
        # Statistics for both disorder_code and diagnostic_category matching
        f.write("\n" + "=" * 100 + "\n")
        f.write("Overall Statistics (both disorder_code and diagnostic_category matching accuracy):\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"{'Filename':<50} {'Common ID':<12} {'Matches':<12} {'Accuracy':<12}\n")
        f.write("-" * 100 + "\n")
        
        for result in all_results:
            both_stats = result['both_stats']
            f.write(f"{result['filename']:<50} ")
            f.write(f"{both_stats['total_common']:<12} ")
            f.write(f"{both_stats['both_match']:<12} ")
            f.write(f"{both_stats['both_match_rate']:.2f}%{'':<8}\n")
        
        # Create diagnostic_category summary table
        f.write("\n" + "=" * 100 + "\n")
        f.write("diagnostic_category Summary Table (Rows: filenames, Columns: category abbreviations):\n")
        f.write("=" * 100 + "\n\n")
        
        # Get category abbreviation mapping
        category_abbrev = get_category_abbreviation()
        
        # Collect all category names
        all_categories = set()
        for result in all_results:
            all_categories.update(result['category_accuracy'].keys())
        
        # Sort category names
        sorted_categories = sorted(all_categories)
        
        # Calculate column width (based on abbreviation names)
        filename_width = max(50, max(len(r['filename']) for r in all_results) + 2)
        # Calculate maximum abbreviation length
        max_abbrev_len = max(len(category_abbrev.get(cat, cat)) for cat in sorted_categories)
        category_width = max(12, max_abbrev_len + 2)  # Width of each category column (display numbers like "XX.XX")
        
        # Write header (first row: category abbreviations)
        f.write(f"{'Filename':<{filename_width}}")
        for category in sorted_categories:
            abbrev = category_abbrev.get(category, category)
            f.write(f"{abbrev:<{category_width}}")
        f.write("\n")
        
        # Write separator line
        separator_length = filename_width + len(sorted_categories) * category_width
        f.write("-" * min(separator_length, 200) + "\n")
        
        # Write each row of data
        for result in all_results:
            f.write(f"{result['filename']:<{filename_width}}")
            for category in sorted_categories:
                if category in result['category_accuracy']:
                    rate = result['category_accuracy'][category]['rate']
                    # Format accuracy, remove percentage sign, only show numbers
                    rate_str = f"{rate:.2f}"
                    f.write(f"{rate_str:<{category_width}}")
                else:
                    f.write(f"{'N/A':<{category_width}}")
            f.write("\n")
        
        # Add description and complete category name list
        f.write("\nNote: Values in the table represent disorder_code matching accuracy for each file in each category (numeric value, percentage sign removed)\n")
        f.write("      N/A indicates that the file has no data in this category\n\n")
        f.write("Category Name Abbreviation Reference (in table column order):\n")
        for i, category in enumerate(sorted_categories, 1):
            abbrev = category_abbrev.get(category, category)
            f.write(f"  {i}. {abbrev}: {category}\n")
        f.write("\n")
        
        # Detailed statistics for each file (including category accuracy)
        f.write("\n" + "=" * 100 + "\n")
        f.write("Detailed Statistics (including diagnostic accuracy for each category):\n")
        f.write("=" * 100 + "\n\n")
        
        for result in all_results:
            f.write(f"\nFile: {result['filename']}\n")
            f.write("-" * 100 + "\n")
            
            code_stats = result['code_stats']
            category_stats = result['category_stats']
            both_stats = result['both_stats']
            f.write(f"Overall Statistics:\n")
            f.write(f"  Result file record count: {result['result_count']}\n")
            f.write(f"  Common ID count: {code_stats['total_common']}\n")
            f.write(f"  GT file record count: {code_stats.get('gt_total', len(gt_data))}\n")
            f.write(f"  disorder_code match: {code_stats['code_match']} / {code_stats.get('gt_total', len(gt_data))} = {code_stats['code_match_rate']:.2f}%\n")
            f.write(f"  diagnostic_category match: {category_stats['category_match']} / {category_stats.get('gt_total', len(gt_data))} = {category_stats['category_match_rate']:.2f}%\n")
            f.write(f"  Both match (disorder_code+diagnostic_category): {both_stats['both_match']} / {both_stats.get('gt_total', len(gt_data))} = {both_stats['both_match_rate']:.2f}%\n")
            
            f.write(f"\nAccuracy by diagnostic_category:\n")
            category_accuracy = result['category_accuracy']
            
            # Sort by accuracy in descending order
            sorted_categories = sorted(category_accuracy.items(), 
                                      key=lambda x: x[1]['rate'], 
                                      reverse=True)
            
            f.write(f"{'Category':<60} {'GT Total':<10} {'Matches':<10} {'Accuracy':<10}\n")
            f.write("-" * 100 + "\n")
            
            for category, stats in sorted_categories:
                # Truncate overly long category names
                category_display = category[:58] + ".." if len(category) > 60 else category
                f.write(f"{category_display:<60} ")
                f.write(f"{stats.get('gt_total', 0):<10} ")  # Display total number of this category in GT
                f.write(f"{stats['match']:<10} ")
                f.write(f"{stats['rate']:.2f}%{'':<6}\n")
            
            f.write("\n")
        
        # Summary statistics (average accuracy across all files)
        f.write("=" * 100 + "\n")
        f.write("Summary Statistics (average accuracy across all files):\n")
        f.write("=" * 100 + "\n\n")
        
        if all_results:
            avg_code_rate = sum(r['code_stats']['code_match_rate'] for r in all_results) / len(all_results)
            avg_category_rate = sum(r['category_stats']['category_match_rate'] for r in all_results) / len(all_results)
            avg_both_rate = sum(r['both_stats']['both_match_rate'] for r in all_results) / len(all_results)
            
            f.write(f"Average disorder_code matching accuracy: {avg_code_rate:.2f}%\n")
            f.write(f"Average diagnostic_category matching accuracy: {avg_category_rate:.2f}%\n")
            f.write(f"Average both match (disorder_code+diagnostic_category) accuracy: {avg_both_rate:.2f}%\n")
            
            # Calculate average accuracy for each category (across all files)
            all_categories = set()
            for result in all_results:
                all_categories.update(result['category_accuracy'].keys())
            
            category_avg_stats = {}
            for category in all_categories:
                rates = []
                totals = []
                for result in all_results:
                    if category in result['category_accuracy']:
                        stats = result['category_accuracy'][category]
                        rates.append(stats['rate'])
                        totals.append(stats.get('gt_total', 0))  # Use gt_total as total
                
                if rates:
                    category_avg_stats[category] = {
                        'avg_rate': sum(rates) / len(rates),
                        'file_count': len(rates),
                        'total_samples': sum(totals)
                    }
            
            if category_avg_stats:
                f.write(f"\nAverage accuracy by diagnostic_category (across all files):\n")
                f.write(f"{'Category':<60} {'File Count':<10} {'Total Samples':<12} {'Avg Accuracy':<12}\n")
                f.write("-" * 100 + "\n")
                
                sorted_avg_categories = sorted(category_avg_stats.items(), 
                                              key=lambda x: x[1]['avg_rate'], 
                                              reverse=True)
                
                for category, stats in sorted_avg_categories:
                    category_display = category[:58] + ".." if len(category) > 60 else category
                    f.write(f"{category_display:<60} ")
                    f.write(f"{stats['file_count']:<10} ")
                    f.write(f"{stats['total_samples']:<12} ")
                    f.write(f"{stats['avg_rate']:.2f}%{'':<8}\n")
    
    print(f"Statistics saved to: {output_file}")
    print("\nProcessing completed!")


if __name__ == "__main__":
    main()

