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
import json

# Define a custom callback for Streamlit integration
class StreamlitProgressCallback(transformers.TrainerCallback):
    """
    Custom callback for tracking training progress in Streamlit.
    
    This provides visual feedback during model training via Streamlit UI elements.
    """
    
    def __init__(self, progress_bar, status_text, log_container, total_steps):
        """
        Initialize the callback with Streamlit UI elements.
        
        Args:
            progress_bar: Streamlit progress bar element
            status_text: Streamlit text element for status updates
            log_container: Streamlit container for detailed logs
            total_steps: Expected total steps in training
        """
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.log_container = log_container
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.loss_history = []
        self.training_loss = 0.0
        self.best_loss = float('inf')
        
        # For log display management
        self.log_history = []
        self.max_logs_to_display = 10  # Limit number of log entries to prevent UI clutter
    
    def on_step_end(self, args, state, control, **kwargs):
        """
        Update progress bar at the end of each step.
        """
        if state.global_step == 0:
            return
        
        self.current_step = state.global_step
        progress = min(float(self.current_step) / self.total_steps, 1.0)
        self.progress_bar.progress(progress)
        
        # Calculate ETA
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            steps_per_second = self.current_step / elapsed_time if elapsed_time > 0 else 0
            remaining_steps = self.total_steps - self.current_step
            
            if steps_per_second > 0:
                eta_seconds = remaining_steps / steps_per_second
                
                if eta_seconds < 60:
                    eta_text = f"{eta_seconds:.0f} seconds"
                elif eta_seconds < 3600:
                    eta_text = f"{eta_seconds/60:.1f} minutes"
                else:
                    eta_text = f"{eta_seconds/3600:.1f} hours"
                    
                self.status_text.text(f"Training step {self.current_step}/{self.total_steps} - " + 
                                    f"Current loss: {self.training_loss:.4f} - " +
                                    f"ETA: {eta_text}")
            else:
                self.status_text.text(f"Training step {self.current_step}/{self.total_steps} - " +
                                    f"Current loss: {self.training_loss:.4f}")
        else:
            self.status_text.text(f"Training step {self.current_step}/{self.total_steps} - Loss: {self.training_loss:.4f}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Display detailed logs.
        """
        if logs:
            # Store loss for history if available
            if "loss" in logs:
                self.training_loss = logs["loss"]
                self.loss_history.append(self.training_loss)
                
                # Track best loss
                if "eval_loss" in logs and logs["eval_loss"] < self.best_loss:
                    self.best_loss = logs["eval_loss"]
            
            # Format log entries more clearly
            log_entries = []
            for k, v in logs.items():
                if isinstance(v, float):
                    log_entries.append(f"{k}: {v:.4f}")
                else:
                    log_entries.append(f"{k}: {v}")
            
            log_str = f"Step {state.global_step}: " + ", ".join(log_entries)
            
            # Add to our log history and limit length
            self.log_history.append(log_str)
            if len(self.log_history) > self.max_logs_to_display:
                self.log_history = self.log_history[-self.max_logs_to_display:]
            
            # Display all kept logs
            self.log_container.text("\n".join(self.log_history))
    
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
        
        if hasattr(state, 'best_model_checkpoint') and state.best_model_checkpoint:
            self.status_text.text(f"Training completed in {time_text} - Best model saved with validation loss: {self.best_loss:.4f}")
        else:
            self.status_text.text(f"Training completed in {time_text} - Final loss: {self.training_loss:.4f}")
        
        # Return loss history for plotting
        return self.loss_history

class GemmaTrainer:
    """
    Handles the fine-tuning of Gemma models with PEFT/LoRA.
    
    This class manages the entire fine-tuning process from model loading to saving.
    """
    
    def __init__(self, config):
        """
        Initialize the trainer with configuration parameters.
        
        Args:
            config: Dictionary containing training configuration parameters
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.output_dir = config["output_dir"]
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """
        Load the model and tokenizer based on configuration.
        """
        progress_bar = st.progress(0.0)
        progress_placeholder = st.empty()
        progress_placeholder.text("Initializing model loading...")
        
        # Load model with memory-efficient settings for quantization
        try:
            quantization_config = None
            device_map = None
            
            progress_bar.progress(0.2)
            progress_placeholder.text("Setting up model configuration...")
            
            # Configure quantization if enabled
            if self.config["use_4bit"]:
                try:
                    import bitsandbytes as bnb
                    
                    # Make sure to handle Mac specific issues
                    if torch.cuda.is_available():
                        # Configure 4-bit quantization
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                        )
                        device_map = "auto"
                    else:
                        # If no CUDA (Mac/CPU), disable quantization
                        st.warning("4-bit quantization requires CUDA GPU. Disabling quantization.")
                        quantization_config = None
                        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            device_map = {"": "mps"}  # Try MPS for Apple Silicon
                        else:
                            device_map = {"": "cpu"}  # Fallback to CPU
                            
                except ImportError:
                    st.warning("bitsandbytes package not found. Disabling 4-bit quantization.")
                    quantization_config = None
                    
            progress_bar.progress(0.4)
            progress_placeholder.text(f"Loading {self.config['model_name']}...")
            
            # Load model with fallbacks for different platforms
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config["model_name"],
                    device_map=device_map,
                    quantization_config=quantization_config,
                    token=os.environ.get("HF_TOKEN")  # Use token if available
                )
            except torch.cuda.OutOfMemoryError:
                # Handle CUDA OOM explicitly
                st.error("GPU ran out of memory while loading the model.")
                st.info("Try enabling 4-bit quantization or using a smaller model.")
                raise
            except ImportError as e:
                if "bitsandbytes" in str(e):
                    st.error("The bitsandbytes package is not correctly installed.")
                    st.info("Try reinstalling with: pip install -U bitsandbytes")
                raise
            except Exception as e:
                # Try fallback to CPU with warning
                st.warning(f"Error loading model with device mapping: {str(e)}")
                st.info("Attempting to fall back to CPU (this will be slow)...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config["model_name"],
                    device_map={"": "cpu"},
                    token=os.environ.get("HF_TOKEN")
                )
                
            progress_bar.progress(0.6)
            progress_placeholder.text("Model loaded, preparing tokenizer...")
            
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
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
        if self.config["use_4bit"] and torch.cuda.is_available():
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
        
        st.success(f"✅ Model and tokenizer loaded successfully!")
        st.info(f"Trainable parameters: {trainable_params:,} ({percentage:.2f}% of total)")
    
    def tokenize_dataset(self, dataset):
        """
        Tokenize the dataset for training.
        
        Args:
            dataset: HuggingFace dataset
        
        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            inputs = self.tokenizer(
                examples["prompt"], 
                padding=False, 
                truncation=True,
                max_length=self.config["max_seq_length"],
                return_tensors=None
            )
            
            # Also tokenize the completions/responses
            outputs = self.tokenizer(
                examples["completion"], 
                padding=False, 
                truncation=True,
                max_length=self.config["max_seq_length"],
                return_tensors=None
            )
            
            # Create full input/output sequences as expected for causal language modeling
            full_inputs = []
            full_labels = []
            
            for i in range(len(inputs["input_ids"])):
                # Concatenate input and output tokens
                full_input = inputs["input_ids"][i] + outputs["input_ids"][i][1:]  # Skip the first token to avoid repetition
                full_label = [-100] * len(inputs["input_ids"][i]) + outputs["input_ids"][i][1:]
                
                # Ensure we don't exceed max length
                if len(full_input) > self.config["max_seq_length"]:
                    full_input = full_input[:self.config["max_seq_length"]]
                    full_label = full_label[:self.config["max_seq_length"]]
                
                full_inputs.append(full_input)
                full_labels.append(full_label)
            
            result = {
                "input_ids": full_inputs,
                "labels": full_labels
            }
            
            return result
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["prompt", "completion"]
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
                save_total_limit=2,  # Keep the best and last checkpoints
                load_best_model_at_end=True,  # Load the best model at the end of training
                metric_for_best_model="eval_loss",  # Use validation loss to determine best model
                greater_is_better=False,  # Lower evaluation loss is better
                report_to="none",  # Disable wandb or other reporting
                run_name=run_name,
                disable_tqdm=True,  # Disable tqdm progress bars, we'll use our own
            )
            
            # Compute total steps for progress bar
            effective_batch_size = (
                training_args.per_device_train_batch_size * 
                training_args.gradient_accumulation_steps * 
                (torch.cuda.device_count() if torch.cuda.is_available() else 1)
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
            
            # Train the model
            self.trainer.train()
            
            # Save the model
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Save training configuration
            with open(os.path.join(self.output_dir, "training_config.json"), "w") as f:
                json.dump(self.config, f, indent=2)
            
            # Get and save the training loss history for visualization
            training_history = streamlit_callback.loss_history
            try:
                with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
                    json.dump({
                        "loss_history": training_history,
                        "best_loss": streamlit_callback.best_loss
                    }, f)
            except:
                pass
            
            return self.output_dir
            
        except torch.cuda.OutOfMemoryError:
            error_msg = "GPU ran out of memory during training."
            st.error(error_msg)
            st.info("Try enabling 4-bit quantization, reducing batch size, or using a smaller model.")
            raise Exception(error_msg)
        
        except ImportError as e:
            error_msg = f"Missing required library: {str(e)}"
            st.error(error_msg)
            st.info("Try reinstalling dependencies with 'pip install -r requirements.txt'")
            raise Exception(error_msg)
        
        except Exception as e:
            error_msg = f"Error during training: {str(e)}"
            st.error(error_msg)
            raise Exception(error_msg)
    
    def create_adapter_zip(self):
        """
        Create a zip file containing the adapter and usage instructions.
        
        Returns:
            tuple: (zip_path, zip_name) - Path to the zip file and its name
        """
        try:
            import zipfile
            import tempfile
            import shutil
            
            # Get base name for the adapter
            adapter_base_name = os.path.basename(self.output_dir)
            zip_name = f"{adapter_base_name}.zip"
            
            # Create a temporary file
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, zip_name)
            
            # Create the zip file
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from the output directory
                for root, dirs, files in os.walk(self.output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(self.output_dir))
                        zipf.write(file_path, arcname)
                
                # Add a README.md file with usage instructions
                readme_content = f"""# Fine-Tuned Gemma Adapter

## What's in this package?
This is a LoRA adapter for the Gemma model, fine-tuned on your custom dataset.
The adapter is significantly smaller than the original model but contains all your customizations.

## How to use your fine-tuned model
You'll need both:
1. The original Gemma model from Hugging Face
2. This adapter that contains your fine-tuning

### Python Example:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model (you need to download this separately)
base_model = "google/gemma-7b"  # or gemma-2b depending on what you used
tokenizer = AutoTokenizer.from_pretrained(base_model)
base_model = AutoModelForCausalLM.from_pretrained(base_model)

# Load your adapter
adapter_path = "./adapter"  # Path to the extracted adapter files
model = PeftModel.from_pretrained(base_model, adapter_path)

# Generate text with your fine-tuned model
prompt = "Write me a summary of..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

## Training Configuration
The adapter was fine-tuned with the following parameters:
- Model: {self.config.get('model_name', 'gemma')}
- Use 4-bit quantization: {self.config.get('use_4bit', 'False')}
- LoRA rank: {self.config.get('lora_r', '64')}
- Batch size: {self.config.get('per_device_train_batch_size', '4')}
- Learning rate: {self.config.get('learning_rate', '2e-4')}
- Epochs: {self.config.get('num_train_epochs', '3')}
"""
                zipf.writestr("README.md", readme_content)
                
                # Add a simple example script
                example_script = """from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model (you need to download this separately)
base_model = "google/gemma-7b"  # or "google/gemma-2b" - use the same model you trained on
tokenizer = AutoTokenizer.from_pretrained(base_model)
base_model = AutoModelForCausalLM.from_pretrained(base_model)

# Load your adapter
adapter_path = "./"  # Path to the extracted adapter files (this directory)
model = PeftModel.from_pretrained(base_model, adapter_path)

# Function to generate responses
def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# Example prompts - replace with relevant examples for your use case
examples = [
    "Write a short story about a robot learning to paint",
    "Explain quantum computing to a 10-year old",
    "What are the key features of a good business plan?"
]

# Generate and print responses
for i, prompt in enumerate(examples):
    print(f"\\nExample {i+1}:\\n{'-'*40}")
    print(f"Prompt: {prompt}")
    print(f"\\nResponse:")
    print(generate_response(prompt))
"""
                zipf.writestr("example.py", example_script)
            
            return zip_path, zip_name
            
        except Exception as e:
            st.error(f"Error creating adapter zip file: {str(e)}")
            return None, None
