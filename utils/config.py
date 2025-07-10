# utils/config.py

"""
Configuration utilities for the L.U.N.A project.
"""

import yaml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    name: str
    num_labels: int
    
@dataclass
class DataConfig:
    raw_dir: str
    processed_dir: str
    train_split: str
    validation_split: str
    test_split: str
    
@dataclass
class TrainConfig:
    output_dir: str
    epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    eval_strategy: str
    save_strategy: str
    best_metric: str

@dataclass
class InferenceConfig:
    threshold: Optional[float] = None
    label_list: List[str] = None
    
@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    inference: InferenceConfig
    max_length: int
    
def load_config(name: str) -> Config:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        Config: An instance of the Config dataclass with loaded values.
    """
    path = f"config/{name}.yaml"
    with open(path, 'r', encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    return Config(
        model=ModelConfig(**config_data['model']),
        data=DataConfig(**config_data['data']),
        train=TrainConfig(
            output_dir=config_data['train']['output_dir'],
            epochs=config_data['train']['epochs'],
            train_batch_size=config_data['train']['train_batch_size'],
            eval_batch_size=config_data['train']['eval_batch_size'],
            learning_rate=float(config_data['train']['learning_rate']),
            eval_strategy=config_data['train']['eval_strategy'],
            save_strategy=config_data['train']['save_strategy'],
            best_metric=config_data['train']['best_metric']
        ),
        inference=InferenceConfig(**config_data['inference']),
        max_length=config_data['max_length']
    )

def load_config_dict(name: str) -> dict:
    path = f"config/{name}.yaml"
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_multitask_config_dict(name: str) -> dict:
    path = f"config/{name}.yaml"
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
