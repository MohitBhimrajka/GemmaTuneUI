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
    Return explanations for the various parameters in the UI.
    
    These are written in simple language to help non-technical users understand
    what each setting does.
    
    Returns:
        Dictionary of parameter names to their explanations
    """
    return {
        "model_name": """
        **Which Gemma model to use:**
        
        • **Gemma 2B**: Smaller, faster, uses less memory. Good for most tasks and smaller GPUs (8GB+).
        
        • **Gemma 7B**: Larger, more capable model. Better quality but requires more GPU memory (12GB+).
        """,
        
        "use_4bit": """
        **Memory-saving mode (QLoRA):**
        
        • **ON** (recommended): Uses advanced memory compression to fit larger models on your GPU.
        
        • **OFF**: Uses more memory but may be slightly more accurate. Only use if you have 24GB+ VRAM.
        """,
        
        "per_device_train_batch_size": """
        **Batch Size:**
        
        How many examples to process at once. Larger values can train faster but need more memory.
        
        • Low memory GPU: Use 1-2
        • Medium memory GPU: Use 2-4
        • High memory GPU (24GB+): Can try 4-8
        """,
        
        "gradient_accumulation_steps": """
        **Gradient Accumulation:**
        
        Simulates a larger batch size without using more memory. Good for small GPUs.
        
        • Example: Batch size 2 with accumulation 4 = Effective batch size of 8
        • Higher numbers = Slower training but more stable
        """,
        
        "learning_rate": """
        **Learning Rate:**
        
        How quickly the model adapts to your examples. 
        
        • Too high: May learn incorrectly
        • Too low: May not learn enough
        • Recommended: 2e-4 (0.0002) for most cases
        """,
        
        "num_train_epochs": """
        **Number of Training Epochs:**
        
        How many times the model goes through your entire dataset.
        
        • Few examples (< 100): Try 5-10 epochs
        • Medium dataset (100-500): Try 3-5 epochs
        • Large dataset (500+): Try 1-3 epochs
        
        More epochs = more customization but risk of overfitting
        """,
        
        "max_steps": """
        **Maximum Training Steps:**
        
        Alternative to epochs - stops after this many steps regardless of dataset size.
        
        • Set to -1 to use epochs instead (recommended)
        • Only change if you want to limit training time
        """,
        
        "warmup_ratio": """
        **Warmup Period:**
        
        The percentage of training used to gradually increase the learning rate.
        
        • Helps model adjust gradually
        • 0.1 = 10% of training used for warmup
        • Recommended: 0.05 to 0.1
        """,
        
        "lora_r": """
        **LoRA Rank:**
        
        Controls how many new patterns the model can learn. Higher = more capacity but uses more memory.
        
        • 8-16: Minimal customization, saves memory
        • 32-64: Balanced (recommended)
        • 128+: Deep customization, needs more memory
        """,
        
        "lora_alpha": """
        **LoRA Alpha:**
        
        Works with LoRA Rank to control adjustment strength. Usually set to 2x the rank.
        
        Default is good for most cases - only change if you know what you're doing.
        """,
        
        "lora_dropout": """
        **LoRA Dropout:**
        
        Helps prevent overfitting (memorizing rather than learning).
        
        • Small datasets: Use higher values (0.1 - 0.2)
        • Large datasets: Use lower values (0.05 - 0.1)
        """,
        
        "max_seq_length": """
        **Maximum Sequence Length:**
        
        The maximum length of text the model can process at once.
        
        • Shorter (256-512): Saves memory, good for short Q&A
        • Longer (1024-2048): Better for longer tasks but uses more memory
        """,
        
        "weight_decay": """
        **Weight Decay:**
        
        Prevents overtraining by keeping weights small.
        
        • Default (0.01) works well for most cases
        • Only change if you're familiar with ML training
        """,
        
        "max_grad_norm": """
        **Maximum Gradient Norm:**
        
        Limits how much the model can change in one step, helps prevent erratic learning.
        
        Default is good for most cases - only change if you know what you're doing.
        """,
        
        "lr_scheduler_type": """
        **Learning Rate Schedule:**
        
        How the learning rate changes during training.
        
        • "Linear": Decreases steadily (good default)
        • "Cosine": Decreases more gradually
        • Others are for specialized cases
        """,
        
        "logging_steps": """
        **Logging Frequency:**
        
        How often to record training progress.
        
        • Lower: More detailed progress updates
        • Higher: Less cluttered logs
        """,
        
        "save_steps": """
        **Checkpoint Frequency:**
        
        How often to save progress during training.
        
        • Lower: More frequent saving but uses more disk space
        • Higher: Less frequent saving
        """,
        
        "target_modules": """
        **Target Modules:**
        
        Which parts of the model to fine-tune. Default targets the key parts.
        
        Don't change unless you know what you're doing.
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
