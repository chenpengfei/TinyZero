"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import os
from datasets import Dataset
from random import randint, seed
from typing import List, Tuple
from tqdm import tqdm
import argparse

def gen_dataset(
    save_path: str,
    num_samples: int,
    num_operands: int = 4,
    max_target: int = 100,
    min_number: int = 1,
    max_number: int = 100,
    seed_value: int = 42,
    upload_to_hf: bool = False,
    hf_repo_id: str = None
) -> List[Tuple]:
    """Generate dataset for countdown task and save to Parquet.
    
    Args:
        num_samples: Number of samples to generate
        num_operands: Number of numbers provided in each sample
        max_target: Maximum value for target number
        min_number: Minimum value for provided numbers
        max_number: Maximum value for provided numbers
        seed_value: Random seed for reproducibility
        save_path: Path to save the generated dataset in Parquet format
        upload_to_hf: Whether to upload the dataset to Hugging Face
        hf_repo_id: Hugging Face repository ID for upload
        
    Returns:
        List of tuples containing (target, numbers)
    """
    seed(seed_value)
    samples = []
    
    for _ in tqdm(range(num_samples)):
        # Generate random target
        target = randint(1, max_target)
        
        # Generate random numbers
        numbers = [randint(min_number, max_number) for _ in range(num_operands)]
        
        samples.append((target, numbers))
    
    # Convert to Hugging Face Dataset and save as Parquet
    dataset = Dataset.from_dict({"target": [s[0] for s in samples], "nums": [s[1] for s in samples]})
    dataset.to_parquet(save_path)
    
    # Upload to Hugging Face if required
    if upload_to_hf and hf_repo_id:
        dataset.push_to_hub(hf_repo_id)
    
    return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/countdown')
    parser.add_argument('--num_samples', type=int, default=490364)
    parser.add_argument('--num_operands', type=int, default=4)
    parser.add_argument('--max_target', type=int, default=100)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--upload_to_hf', action='store_true')
    parser.add_argument('--hf_repo_id', type=str, default=None)

    args = parser.parse_args()

    # Generate and save dataset
    gen_dataset(
        save_path=os.path.join(args.local_dir, 'train-00000-of-00001.parquet'),
        num_samples=args.num_samples,
        num_operands=args.num_operands,
        max_target=args.max_target,
        min_number=args.min_number,
        max_number=args.max_number,
        upload_to_hf=args.upload_to_hf,
        hf_repo_id=args.hf_repo_id
    )
