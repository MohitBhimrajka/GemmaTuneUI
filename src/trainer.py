"""
Core fine-tuning logic for Gemma models using PEFT (Parameter-Efficient Fine-Tuning).
"""

import os
import torch
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
import bitsandbytes as bnb
import streamlit as st
import shutil
import tempfile
import zipfile
from datetime import datetime

# Define a custom callback for Streamlit integration
class StreamlitProgressCallback(transformers.TrainerCallback):
    """
    A callback to display training progress in Streamlit.
    """
    def __init__(self, progress_bar, status_text, log_container, total_steps):
        """
        Initialize the callback with Streamlit elements.
        
        Args:
            progress_bar: Streamlit progress bar element
            status_text: Streamlit text element for status updates
            log_container: Streamlit container for detailed logs
            total_steps: Total number of training steps
        """
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.log_container = log_container
        self.total_steps = total_steps
        self.current_step = 0
        self.training_loss = None
    
    def on_step_end(self, args, state, control, **kwargs):
        """
        Update progress bar and status text on each step.
        """
        self.current_step = state.global_step
        if self.total_steps > 0:
            self.progress_bar.progress(min(1.0, self.current_step / self.total_steps))
        
        # Only update every 10 steps to avoid overwhelming the UI
        if self.current_step % 10 == 0 or self.current_step == 1:
            # Get loss if available
            loss_text = ""
            if state.log_history:
                logs = [log for log in state.log_history if 'loss' in log]
                if logs:
                    self.training_loss = logs[-1]['loss']
                    loss_text = f" | Loss: {self.training_loss:.4f}"
            
            self.status_text.text(f"Step {self.current_step}/{self.total_steps}{loss_text}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Display detailed logs.
        """
        if logs:
            log_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                for k, v in logs.items()])
            self.log_container.text(f"Step {self.current_step}: {log_str}")

class GemmaTrainer:
    """
    Class for fine-tuning Gemma models with QLoRA.
    """
    
    def __init__(self, config):
        """
        Initialize the trainer with the given configuration.
        
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.device_map = "auto"
        self.output_dir = config["output_dir"]
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """
        Load the model and tokenizer.
        """
        st.text("Loading model and tokenizer...")
        
        # Determine model loading parameters based on config
        model_kwargs = {}
        if self.config["use_4bit"]:
            model_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "device_map": self.device_map
            })
        
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"],
                torch_dtype=torch.float16,
                **model_kwargs
            )
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
            
            # Set padding on the right
            self.tokenizer.padding_side = "right"
            
            # Force the tokenizer to add EOS token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise Exception(f"Failed to load tokenizer: {str(e)}")
        
        # If using 4-bit quantization, prepare the model
        if self.config["use_4bit"]:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            target_modules=self.config["target_modules"],
            lora_dropout=self.config["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        st.text(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of {total_params:,} total)")
        
        return self.model, self.tokenizer
    
    def tokenize_dataset(self, dataset):
        """
        Tokenize the dataset.
        
        Args:
            dataset: HuggingFace dataset containing a 'text' column
            
        Returns:
            tokenized_dataset: Tokenized dataset
        """
        # Set the maximum length to tokenize
        max_length = 512  # Default, adjust based on your data and model
        
        def tokenize_function(examples):
            # Tokenize examples
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            return result
        
        # Apply tokenization to dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_dataset
    
    def train(self, dataset, progress_bar, status_text, log_container):
        """
        Train the model.
        
        Args:
            dataset: HuggingFace dataset
            progress_bar: Streamlit progress bar element
            status_text: Streamlit text element for status updates
            log_container: Streamlit container for detailed logs
            
        Returns:
            output_dir: Directory containing saved model
        """
        # Load model and tokenizer if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # Tokenize dataset
        tokenized_dataset = self.tokenize_dataset(dataset)
        
        # Check if dataset is empty after tokenization
        if len(tokenized_dataset) == 0:
            raise Exception("Tokenized dataset is empty. Please check your data.")
        
        # Split dataset into train and validation sets (80/20)
        split_datasets = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split_datasets["train"]
        eval_dataset = split_datasets["test"]
        
        # Set up training arguments
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"gemma_lora_{current_time}"
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            learning_rate=self.config["learning_rate"],
            lr_scheduler_type=self.config["lr_scheduler_type"],
            warmup_ratio=self.config["warmup_ratio"],
            weight_decay=self.config["weight_decay"],
            max_grad_norm=self.config["max_grad_norm"],
            num_train_epochs=self.config["num_train_epochs"],
            max_steps=self.config["max_steps"],
            logging_steps=self.config["logging_steps"],
            save_steps=self.config["save_steps"],
            evaluation_strategy="steps" if len(eval_dataset) > 0 else "no",
            eval_steps=self.config["save_steps"] if len(eval_dataset) > 0 else None,
            save_total_limit=1,  # Only keep the most recent checkpoint
            load_best_model_at_end=False,
            report_to="none",  # Disable wandb or other reporting
            run_name=run_name,
            disable_tqdm=True,  # Disable tqdm progress bars, we'll use our own
        )
        
        # Compute total steps for progress bar
        total_steps = (
            len(train_dataset) // 
            (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * 
            training_args.num_train_epochs
        )
        
        # Initialize our custom Streamlit callback
        streamlit_callback = StreamlitProgressCallback(
            progress_bar=progress_bar,
            status_text=status_text,
            log_container=log_container,
            total_steps=total_steps
        )
        
        # Set up data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False  # Not using masked language modeling
        )
        
        # Configure Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
            data_collator=data_collator,
            callbacks=[streamlit_callback]
        )
        
        # Start training
        try:
            self.trainer.train()
            
            # Save the final model
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Return the output directory
            return self.output_dir
            
        except Exception as e:
            raise Exception(f"Training error: {str(e)}")
    
    def create_adapter_zip(self):
        """
        Create a downloadable zip file containing the adapter weights.
        
        Returns:
            tuple: (zip_file_path, zip_file_name)
        """
        try:
            temp_dir = tempfile.mkdtemp()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_file_name = f"gemma_lora_adapter_{timestamp}.zip"
            zip_file_path = os.path.join(temp_dir, zip_file_name)
            
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from the output directory
                for root, _, files in os.walk(self.output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.output_dir)
                        zipf.write(file_path, arcname)
                
                # Add a simple README
                readme_content = f"""# Gemma LoRA Adapter

This adapter was fine-tuned from the {self.config['model_name']} model using QLoRA.

## Loading Instructions

To use this adapter with the base model:

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{self.config['model_name']}")
tokenizer = AutoTokenizer.from_pretrained("{self.config['model_name']}")

# Load the LoRA adapter
config = PeftConfig.from_pretrained("./adapter_directory")
model = PeftModel.from_pretrained(model, "./adapter_directory")

# Use the model
input_text = "Your prompt here"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Configuration
{str(self.config)}
"""
                zipf.writestr("README.md", readme_content)
            
            return zip_file_path, zip_file_name
            
        except Exception as e:
            st.error(f"Failed to create zip file: {str(e)}")
            return None, None
