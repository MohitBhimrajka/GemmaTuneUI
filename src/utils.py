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
    Using simpler language for non-technical users.
    """
    return {
        # Model options
        "model_name": """
        Choose which Gemma model to customize:
        
        • 2B models are smaller and train faster - good for beginners or limited GPUs
        • 7B models potentially give better results but need more powerful GPUs
        • Models with 'it' are 'instruction-tuned' - already better at following directions
        """,
        
        # Quantization options
        "use_4bit": """
        Makes the model smaller to fit in your GPU's memory.
        
        Keep this ON unless you have a very powerful GPU (24GB+ memory).
        It slightly reduces precision but makes training possible on most GPUs.
        """,
        
        # Training parameters
        "num_train_epochs": """
        How many times the AI will see your entire dataset during training.
        
        • Lower (1-2): Faster training, might not learn as well
        • Medium (3-5): Better learning, takes longer
        • Higher (6+): Usually not needed unless you have lots of data
        """,
        
        "per_device_train_batch_size": """
        How many examples the AI processes at once.
        
        • Smaller values (1-2): Use less GPU memory but train slower
        • Larger values (4+): May improve learning but need more memory
        
        If you get 'out of memory' errors, lower this value.
        """,
        
        "gradient_accumulation_steps": """
        A way to simulate larger batch sizes on limited memory.
        
        Think of it as collecting multiple small batches before doing an update.
        Higher values = more stable training but slower.
        """,
        
        "learning_rate": """
        How quickly the AI adapts to your examples.
        
        • Too low: Learning happens too slowly
        • Too high: Learning becomes unstable
        
        The default value works well for most cases.
        """,
        
        "lr_scheduler_type": """
        Controls how the learning rate changes during training.
        
        'Cosine' gradually reduces the learning rate, helping the model fine-tune
        more precisely as training progresses.
        """,
        
        "warmup_ratio": """
        Percentage of training spent gradually increasing the learning rate.
        
        This helps the model start learning smoothly. The default works well.
        """,
        
        "weight_decay": """
        Helps prevent the model from 'memorizing' your examples.
        
        It encourages the model to learn general patterns rather than
        specific details. The default works well for most cases.
        """,
        
        "max_grad_norm": """
        Limits how much the model can change in a single update.
        
        This provides training stability. You rarely need to adjust this.
        """,
        
        # LoRA parameters
        "lora_r": """
        Controls how much the model can learn new things.
        
        • Lower values (4-8): Work well for small datasets, use less memory
        • Higher values (16-32): Can learn more complex patterns but risk
          memorizing examples and use more memory
        """,
        
        "lora_alpha": """
        Scaling factor for LoRA. Usually set to 2× the LoRA rank.
        
        Controls the strength of updates. The default works well for most cases.
        """,
        
        "lora_dropout": """
        Helps prevent memorization by randomly ignoring some connections during training.
        
        • Lower values (0.0-0.1): Good for larger datasets
        • Higher values (0.1-0.3): Help with small datasets
        """,
        
        "target_modules": """
        Which parts of the model to fine-tune with LoRA.
        
        These are the attention layers and feed-forward networks.
        The default settings work well for most cases.
        """,
    }

def get_column_name_variations():
    """
    Returns lists of common column name variations for prompts and completions.
    Used for auto-detecting columns in user datasets.
    """
    return {
        "prompt_columns": ["prompt", "instruction", "input", "question", "context", "user_input", "user", "query", "human", "request", "task"],
        "completion_columns": ["completion", "response", "output", "answer", "assistant", "model_output", "model", "result", "generated", "ai", "bot", "gpt", "assistant_response"],
        "text_columns": ["text", "content", "message", "full_text", "conversation", "dialogue", "data", "sample"]
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
