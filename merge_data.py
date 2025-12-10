#!/usr/bin/env python3
"""Merge puzzle and SF position data into single training file."""

import json
import random
import argparse

def merge_files(puzzle_file, sf_file, output_file):
    """Merge and shuffle training data."""
    print(f"Loading {puzzle_file}...")
    with open(puzzle_file, 'r') as f:
        puzzles = [json.loads(line) for line in f]
    print(f"  Loaded {len(puzzles):,} puzzles")
    
    print(f"Loading {sf_file}...")
    with open(sf_file, 'r') as f:
        sf_positions = [json.loads(line) for line in f]
    print(f"  Loaded {len(sf_positions):,} SF positions")
    
    # Combine
    all_data = puzzles + sf_positions
    print(f"\nTotal: {len(all_data):,} examples")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_data)
    
    # Remove metadata for cleaner training file
    clean_data = []
    for item in all_data:
        clean_data.append({"messages": item["messages"]})
    
    # Write
    print(f"Writing to {output_file}...")
    with open(output_file, 'w') as f:
        for item in clean_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Done! Created {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzles", default="puzzles_labeled.jsonl")
    parser.add_argument("--sf", default="sf_labeled.jsonl")
    parser.add_argument("--output", default="train_data_280k.jsonl")
    args = parser.parse_args()
    
    merge_files(args.puzzles, args.sf, args.output)
