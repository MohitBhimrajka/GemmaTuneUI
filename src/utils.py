"""
Utility functions for GemmaTuneUI, including default configurations and explanations.
"""

def get_default_config():
    """
    Returns a dictionary of default configurations for model fine-tuning.
    """
    return {
        # Model options
        "model_name": "google/gemma-2b",
        "model_options": ["google/gemma-2b", "google/gemma-2b-it", "google/gemma-7b", "google/gemma-7b-it"],
        
        # Quantization options
        "use_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        
        # Training parameters
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.001,
        "max_grad_norm": 0.3,
        "max_steps": -1,  # -1 means train for num_train_epochs
        "save_steps": 100,
        "logging_steps": 10,
        
        # LoRA parameters
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # Output directory
        "output_dir": "./lora-gemma-output",
    }

def get_parameter_explanations():
    """
    Returns a dictionary of explanatory text snippets for hyperparameters.
    """
    return {
        # Model options
        "model_name": "Select which version of Google's Gemma model to fine-tune. Models with 'it' are instruction-tuned versions, generally better for following directions.",
        
        # Quantization options
        "use_4bit": "Reduces the precision of model numbers from 32-bit to 4-bit. This drastically reduces memory usage (making it possible to run on consumer GPUs) with minimal impact on quality.",
        
        # Training parameters
        "num_train_epochs": "How many times the AI will see your entire dataset during training. More epochs can help learning but risks overfitting (memorizing instead of generalizing).",
        "per_device_train_batch_size": "How many examples to process at once. Larger batches give more stable training but use more GPU memory.",
        "gradient_accumulation_steps": "A technique to simulate larger batch sizes on limited memory GPUs. If your batch size is 2 and this is 4, it's like having a batch size of 8.",
        "learning_rate": "How quickly the model adapts to your data. Too high: unstable training. Too low: slow progress. Think of it as the 'step size' for learning.",
        "lr_scheduler_type": "Controls how the learning rate changes during training. 'Cosine' gradually reduces the learning rate, which often works well.",
        "warmup_ratio": "Percentage of training spent gradually increasing the learning rate before decreasing it. Helps stabilize early training.",
        "weight_decay": "A form of regularization that prevents parameter values from growing too large, which helps prevent overfitting.",
        "max_grad_norm": "Limits how much the model can change in a single update, providing training stability.",
        
        # LoRA parameters
        "lora_r": "The 'rank' of LoRA adapters. Higher values (16, 32, etc.) give the model more capacity to learn but use more memory and risk overfitting on small datasets.",
        "lora_alpha": "Scaling factor for LoRA. Usually set to 2x the LoRA rank. Controls the magnitude of updates.",
        "lora_dropout": "Randomly drops connections during training to prevent overfitting. Higher values (0.1, 0.2) provide more regularization for small datasets.",
        "target_modules": "Which parts of the model to fine-tune with LoRA. These represent the attention mechanism and feed-forward parts of the model.",
    }

def get_column_name_variations():
    """
    Returns lists of common column name variations for prompts and completions.
    Used for auto-detecting columns in user datasets.
    """
    return {
        "prompt_columns": ["prompt", "instruction", "input", "question", "context", "user_input"],
        "completion_columns": ["completion", "response", "output", "answer", "assistant", "model_output"],
        "text_columns": ["text", "content", "message"]
    }

def get_gemma_chat_template():
    """
    Returns the chat template format for Gemma models.
    """
    return "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{completion}<end_of_turn>"

def format_size(size_bytes):
    """
    Format file size from bytes to human-readable format.
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0 or unit == 'GB':
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
