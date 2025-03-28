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
import gc  # For garbage collection
import time

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
        self.start_time = time.time()
        self.loss_history = []
    
    def on_step_end(self, args, state, control, **kwargs):
        """
        Update progress bar and status text on each step.
        """
        self.current_step = state.global_step
        if self.total_steps > 0:
            progress = min(1.0, self.current_step / self.total_steps)
            self.progress_bar.progress(progress)
        
        # Only update every few steps to avoid overwhelming the UI
        if self.current_step % 5 == 0 or self.current_step == 1:
            # Get loss if available
            loss_text = ""
            if state.log_history:
                logs = [log for log in state.log_history if 'loss' in log]
                if logs:
                    self.training_loss = logs[-1]['loss']
                    self.loss_history.append(self.training_loss)
                    loss_text = f" | Loss: {self.training_loss:.4f}"
            
            # Calculate and show ETA
            elapsed_time = time.time() - self.start_time
            if self.current_step > 0 and self.total_steps > 0:
                time_per_step = elapsed_time / self.current_step
                remaining_steps = self.total_steps - self.current_step
                eta_seconds = remaining_steps * time_per_step
                
                # Format time nicely
                if eta_seconds < 60:
                    eta_text = f"{eta_seconds:.0f} seconds"
                elif eta_seconds < 3600:
                    eta_text = f"{eta_seconds/60:.1f} minutes"
                else:
                    eta_text = f"{eta_seconds/3600:.1f} hours"
                
                self.status_text.text(f"Step {self.current_step}/{self.total_steps}{loss_text} | ETA: {eta_text}")
            else:
                self.status_text.text(f"Step {self.current_step}/{self.total_steps}{loss_text}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Display detailed logs.
        """
        if logs:
            # Format log entries more clearly
            log_entries = []
            for k, v in logs.items():
                if isinstance(v, float):
                    log_entries.append(f"{k}: {v:.4f}")
                else:
                    log_entries.append(f"{k}: {v}")
            
            log_str = ", ".join(log_entries)
            self.log_container.text(f"Step {self.current_step}: {log_str}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """
        Display start of training message.
        """
        self.start_time = time.time()
        self.status_text.text("Training started. Preparing first batch...")
        
    def on_train_end(self, args, state, control, **kwargs):
        """
        Display end of training message and summary.
        """
        total_time = time.time() - self.start_time
        
        # Format time nicely
        if total_time < 60:
            time_text = f"{total_time:.0f} seconds"
        elif total_time < 3600:
            time_text = f"{total_time/60:.1f} minutes"
        else:
            time_text = f"{total_time/3600:.1f} hours"
            
        self.status_text.text(f"Training completed in {time_text} - Final loss: {self.training_loss:.4f}")
        
        # Return loss history for plotting
        return self.loss_history

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
        # First check if CUDA is available
        if not torch.cuda.is_available():
            st.error("❌ CUDA not detected. Gemma requires a CUDA-enabled GPU for fine-tuning.")
            raise RuntimeError("No CUDA GPUs are available. This application requires GPU acceleration.")
        
        # Check GPU memory for warnings
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        model_size = "7B" if "7b" in self.config["model_name"].lower() else "2B"
        
        if model_size == "7B" and gpu_memory < 16 and not self.config["use_4bit"]:
            st.warning("""
            ⚠️ Warning: You're trying to load a 7B model without quantization on a GPU with less than 16GB memory.
            This will likely cause out-of-memory errors. Enabling 4-bit quantization is strongly recommended.
            """)
        elif model_size == "7B" and gpu_memory < 10:
            st.warning("""
            ⚠️ Warning: Loading a 7B model on a GPU with less than 10GB may be challenging even with quantization.
            If you encounter memory errors, try the 2B model instead.
            """)
            
        st.text(f"Loading {model_size} model and tokenizer - this may take a minute...")
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        
        # Determine model loading parameters based on config
        model_kwargs = {}
        if self.config["use_4bit"]:
            progress_bar.progress(0.2)
            progress_placeholder.text("Setting up 4-bit quantization...")
            model_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "device_map": self.device_map
            })
        
        # Load model with proper error handling and feedback
        try:
            progress_placeholder.text(f"Downloading and loading {model_size} Gemma model...")
            progress_bar.progress(0.4)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"],
                torch_dtype=torch.float16,
                **model_kwargs
            )
            
            progress_bar.progress(0.7)
            progress_placeholder.text("Loading tokenizer...")
            
        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                st.error("""
                ❌ GPU out of memory while loading the model!
                
                Suggestions:
                1. Enable 4-bit quantization in the sidebar
                2. Try a smaller model (Gemma 2B instead of 7B)
                3. Close other GPU-intensive applications
                """)
            elif "not found" in error_msg.lower() or "404" in error_msg:
                st.error(f"""
                ❌ Model '{self.config['model_name']}' not found!
                
                Please check the model name or your internet connection.
                """)
            else:
                st.error(f"Failed to load model: {error_msg}")
            
            raise Exception(f"Failed to load model: {str(e)}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
            
            # Set padding on the right
            self.tokenizer.padding_side = "right"
            
            # Force the tokenizer to add EOS token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            progress_bar.progress(0.8)
            progress_placeholder.text("Preparing model for fine-tuning...")
            
        except Exception as e:
            st.error(f"Failed to load tokenizer: {str(e)}")
            raise Exception(f"Failed to load tokenizer: {str(e)}")
        
        # If using 4-bit quantization, prepare the model
        if self.config["use_4bit"]:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        progress_bar.progress(0.9)
        progress_placeholder.text("Configuring LoRA adapters...")
        
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
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0
        
        progress_bar.progress(1.0)
        progress_placeholder.empty()
        
        st.success(f"""
        ✅ Model loaded successfully!
        
        Model: {self.config['model_name']}
        Trainable parameters: {trainable_params:,} ({percentage:.2f}% of total)
        Quantization: {"Enabled (4-bit)" if self.config["use_4bit"] else "Disabled"}
        """)
        
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
        
        st.text("Tokenizing your dataset...")
        tokenize_progress = st.progress(0)
        
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
        
        tokenize_progress.progress(1.0)
        st.success(f"✅ Dataset tokenized successfully with {len(tokenized_dataset)} examples!")
        
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
        try:
            # Load model and tokenizer if not already loaded
            if self.model is None or self.tokenizer is None:
                self.load_model_and_tokenizer()
            
            # Tokenize dataset
            tokenized_dataset = self.tokenize_dataset(dataset)
            
            # Check if dataset is empty after tokenization
            if len(tokenized_dataset) == 0:
                raise Exception("Tokenized dataset is empty. Please check your data.")
            
            # Split dataset into train and validation sets (80/20)
            st.text("Splitting dataset into training and validation sets (80/20)...")
            split_datasets = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
            train_dataset = split_datasets["train"]
            eval_dataset = split_datasets["test"]
            
            st.success(f"✅ Training set: {len(train_dataset)} examples, Validation set: {len(eval_dataset)} examples")
            
            # Set up training arguments
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"gemma_lora_{current_time}"
            
            st.text("Configuring training parameters...")
            
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
            effective_batch_size = (
                training_args.per_device_train_batch_size * 
                training_args.gradient_accumulation_steps * 
                torch.cuda.device_count()
            )
            
            total_steps = (
                len(train_dataset) // effective_batch_size * 
                training_args.num_train_epochs
            )
            
            if total_steps == 0:
                # In case the dataset is smaller than the batch size
                total_steps = len(train_dataset) * training_args.num_train_epochs
            
            st.markdown(f"""
            **Training Configuration:**
            - Epochs: {training_args.num_train_epochs}
            - Batch size: {training_args.per_device_train_batch_size}
            - Learning rate: {training_args.learning_rate}
            - Gradient accumulation steps: {training_args.gradient_accumulation_steps}
            - Total training steps: ~{total_steps}
            """)
            
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
            st.text("Starting the training process...")
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
                train_result = self.trainer.train()
                
                # Save the final model and get training loss history
                st.text("Saving your fine-tuned model...")
                self.model.save_pretrained(self.output_dir)
                self.tokenizer.save_pretrained(self.output_dir)
                
                # Store training loss history in session state
                if hasattr(streamlit_callback, 'loss_history'):
                    st.session_state.training_history = streamlit_callback.loss_history
                
                # Return the output directory
                return self.output_dir
                
            except Exception as e:
                error_msg = str(e)
                
                if "out of memory" in error_msg.lower():
                    raise Exception(f"""CUDA out of memory during training. Try these solutions:
                    1. Reduce batch size (current: {self.config['per_device_train_batch_size']})
                    2. Enable 4-bit quantization if not already enabled
                    3. Use a smaller model (e.g., 2B instead of 7B)
                    4. Free up GPU memory by closing other applications
                    
                    Error details: {error_msg}
                    """)
                else:
                    raise Exception(f"Training error: {error_msg}")
                
        except Exception as e:
            # Clean up memory to avoid lingering tensors
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            if self.trainer is not None:
                del self.trainer
            
            torch.cuda.empty_cache()
            gc.collect()
            
            raise Exception(str(e))
    
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
            
            st.text("Creating downloadable zip file of your adapter...")
            
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from the output directory
                for root, _, files in os.walk(self.output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.output_dir)
                        zipf.write(file_path, arcname)
                
                # Add a simple README with clearer instructions
                readme_content = f"""# Gemma Fine-Tuned Adapter

This adapter was fine-tuned from the {self.config['model_name']} model using QLoRA.

## 🚀 How to Use Your Personalized AI

### Option 1: Simple Python Script

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load the base model (must match what you fine-tuned)
model = AutoModelForCausalLM.from_pretrained(
    "{self.config['model_name']}",
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{self.config['model_name']}")

# 2. Load your adapter
# First extract this zip file to a folder, then:
model = PeftModel.from_pretrained(model, "./adapter_directory")

# 3. Use your personalized model
prompt = "Write a short poem about the ocean"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_length=200,
    temperature=0.7
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Option 2: Interactive Experience with Gradio

```python
import gradio as gr
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer (same as above)
model_name = "{self.config['model_name']}"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = PeftModel.from_pretrained(model, "./adapter_directory")

def generate_response(input_text, max_length=200, temperature=0.7):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create a simple Gradio interface
demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(lines=3, placeholder="Enter your prompt here..."),
        gr.Slider(minimum=50, maximum=500, value=200, step=10, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature")
    ],
    outputs="text",
    title="Your Fine-Tuned Gemma Model",
    description="Enter a prompt to get a response from your personalized AI"
)

demo.launch()
```

## Training Configuration
```
{str({k: v for k, v in self.config.items() if k != 'target_modules'})}
```

## Need Help?
- Check the [PEFT documentation](https://huggingface.co/docs/peft)
- Explore [Gemma examples](https://huggingface.co/google/gemma-7b)
"""
                zipf.writestr("README.md", readme_content)
                
                # Add a simple example Python script
                example_script = """
# example.py - Simple script to use your fine-tuned adapter

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Change these values as needed
MODEL_NAME = "google/gemma-2b"  # Must match what you fine-tuned
ADAPTER_PATH = "./adapter"  # Path to extracted adapter directory
PROMPT = "Write a short story about a robot who wants to be human"

# Load base model and tokenizer
print(f"Loading base model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load your adapter
print(f"Loading adapter from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# Generate a response
print(f"Generating response to: {PROMPT}")
inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_length=500,
    temperature=0.7,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "-"*50)
print("RESPONSE:")
print("-"*50)
print(response)
"""
                zipf.writestr("example.py", example_script)
            
            st.success("✅ Adapter zip file created successfully!")
            return zip_file_path, zip_file_name
            
        except Exception as e:
            st.error(f"Failed to create zip file: {str(e)}")
            return None, None
