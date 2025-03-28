"""
Helper functions to build parts of the Streamlit UI.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from src.data_handler import get_demo_dataset
import base64
from io import BytesIO

def render_sidebar(default_config, explanations):
    """
    Render the sidebar with configuration options.
    
    Args:
        default_config: Dictionary of default configuration values
        explanations: Dictionary of explanatory text for parameters
        
    Returns:
        dict: Selected configuration
    """
    st.sidebar.title("Training Configuration")
    
    # Create a copy of the default config to modify
    config = default_config.copy()
    
    # Model section with improved header and explanation
    st.sidebar.header("📚 Model Selection")
    st.sidebar.markdown("Choose which Gemma model to customize:")
    
    config["model_name"] = st.sidebar.selectbox(
        "Gemma Model",
        options=default_config["model_options"],
        index=default_config["model_options"].index(default_config["model_name"]),
        help=explanations["model_name"]
    )
    
    # Add a note about model sizes for first-time users
    if "2b" in config["model_name"].lower():
        st.sidebar.caption("2B models are smaller and faster to train, great for beginners.")
    elif "7b" in config["model_name"].lower():
        st.sidebar.caption("7B models are larger and potentially more capable, but require more GPU memory.")
    
    # Quantization section with clearer explanation
    st.sidebar.header("🧠 Memory Optimization")
    config["use_4bit"] = st.sidebar.checkbox(
        "Use 4-bit Quantization (QLoRA)",
        value=default_config["use_4bit"],
        help=explanations["use_4bit"]
    )
    
    if config["use_4bit"]:
        st.sidebar.caption("✅ Recommended: QLoRA drastically reduces memory usage with minimal quality impact.")
    else:
        st.sidebar.caption("⚠️ Warning: Disabling quantization requires significantly more GPU memory.")
    
    # Training parameters section with improved grouping and explanations
    st.sidebar.header("⚙️ Training Parameters")
    
    # Use a column layout to group related parameters
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        config["num_train_epochs"] = st.number_input(
            "Number of Epochs",
            min_value=1,
            max_value=10,
            value=default_config["num_train_epochs"],
            help=explanations["num_train_epochs"]
        )
        st.caption("How many times to process your data")
    
    with col2:
        config["learning_rate"] = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-3,
            value=default_config["learning_rate"],
            format="%.1e",
            help=explanations["learning_rate"]
        )
        st.caption("How quickly the AI adapts")
    
    # Batch size with memory warning
    config["per_device_train_batch_size"] = st.sidebar.number_input(
        "Batch Size",
        min_value=1,
        max_value=8,
        value=default_config["per_device_train_batch_size"],
        help=explanations["per_device_train_batch_size"]
    )
    
    if config["per_device_train_batch_size"] > 2:
        st.sidebar.caption("⚠️ Larger batch sizes use more GPU memory. Start smaller if you encounter errors.")
    else:
        st.sidebar.caption("✅ Smaller batch sizes use less GPU memory but may train slightly slower.")
    
    config["gradient_accumulation_steps"] = st.sidebar.number_input(
        "Gradient Accumulation Steps",
        min_value=1,
        max_value=16,
        value=default_config["gradient_accumulation_steps"],
        help=explanations["gradient_accumulation_steps"]
    )
    
    # LoRA parameters section - more beginner-friendly explanation
    st.sidebar.header("🔧 Fine-Tuning Settings")
    st.sidebar.markdown("**LoRA Parameters** (how the model learns)")
    
    with st.sidebar.expander("Advanced Settings", expanded=False):
        st.caption("These settings control how much the model can change during training. Default values work well for most cases.")
        
        config["lora_r"] = st.slider(
            "LoRA Rank (r)",
            min_value=1,
            max_value=64,
            value=default_config["lora_r"],
            help=explanations["lora_r"]
        )
        
        config["lora_alpha"] = st.slider(
            "LoRA Alpha",
            min_value=1,
            max_value=128,
            value=default_config["lora_alpha"],
            help=explanations["lora_alpha"]
        )
        
        config["lora_dropout"] = st.slider(
            "LoRA Dropout",
            min_value=0.0,
            max_value=0.9,
            value=default_config["lora_dropout"],
            step=0.05,
            format="%.2f",
            help=explanations["lora_dropout"]
        )
    
    # Add a GPU memory indicator with clearer information
    st.sidebar.markdown("### 📊 Hardware Requirements")
    
    if config["model_name"].startswith("google/gemma-2b"):
        gpu_mem_needed = "~8GB VRAM" if config["use_4bit"] else "~14GB VRAM"
        model_size = "2B"
    else:  # 7B model
        gpu_mem_needed = "~14GB VRAM" if config["use_4bit"] else "~28GB VRAM"
        model_size = "7B"
    
    st.sidebar.info(f"""
    **Estimated GPU Memory Needed:** {gpu_mem_needed}
    
    **Recommended GPUs for {model_size} model:**
    {" RTX 3060+ " if model_size == "2B" else " RTX 3080/3090+ "}
    """)
    
    return config

def display_training_progress(title="Training Progress"):
    """
    Create and display placeholders for training progress.
    
    Args:
        title: Title for the training progress section
        
    Returns:
        tuple: (progress_bar, status_text, log_container)
    """
    if title:
        st.subheader(title)
    
    # Create a more descriptive progress layout
    progress_container = st.container()
    progress_bar = progress_container.progress(0)
    
    # Status with more context
    status_container = st.empty()
    status_text = status_container.text("Preparing to train...")
    
    # Container for detailed logs with explanation
    st.caption("Detailed training logs will appear below:")
    log_container = st.empty()
    
    return progress_bar, status_text, log_container

def display_results(output_dir, training_history=None):
    """
    Display training results and provide download option.
    
    Args:
        output_dir: Directory containing training results
        training_history: Optional history of loss values for plotting
    """
    st.markdown("### Your Custom AI Adapter is Ready!")
    
    # Prepare for download - create a zip file with the adapter and instructions
    try:
        import os
        import zipfile
        import tempfile
        import shutil
        
        # Create a temporary file for the zip
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip_file:
            temp_zip_path = temp_zip_file.name
        
        # Create the zip file
        with zipfile.ZipFile(temp_zip_path, 'w') as zipf:
            # Walk through the output directory and add all files
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(output_dir))
                    zipf.write(file_path, arcname)
            
            # Add a README file with usage instructions
            readme_content = """# Your Custom Gemma Adapter

## What's in this package?
This package contains a LoRA adapter for Google's Gemma model that has been fine-tuned on your data.
The adapter is a small file that modifies the base Gemma model to behave according to your examples.

## How to use your adapter
1. You'll need the base Gemma model from Hugging Face
2. Load both the base model and this adapter to get your customized AI

### Python code example:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load the base model
model_name = "google/gemma-7b"  # or "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load your custom adapter
adapter_path = "./gemma_adapter"  # Path to the extracted adapter files
model = PeftModel.from_pretrained(model, adapter_path)

# Generate text with your custom model
prompt = "Write a summary of the latest quarterly report"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Important notes
- This is NOT the complete model, just an adapter (a few MB vs. several GB)
- You still need to download the base Gemma model separately
- The adapter works best on similar tasks to your training data
"""
            zipf.writestr("README.md", readme_content)
            
            # Add a sample Python script to use the adapter
            sample_code = """from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load the base model
model_name = "google/gemma-7b"  # or "google/gemma-2b" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model)

# Load your custom adapter
adapter_path = "./gemma_adapter"  # Path to the extracted adapter files
model = PeftModel.from_pretrained(model, adapter_path)

# Generate text with your custom model
def generate_response(prompt, max_length=500):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Write a summary of the latest quarterly report"
response = generate_response(prompt)
print(response)
"""
            zipf.writestr("use_your_adapter.py", sample_code)
        
        # Provide download link
        with open(temp_zip_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Your Custom AI Adapter",
                data=f,
                file_name="gemma_custom_adapter.zip",
                mime="application/zip",
                help="Download your fine-tuned adapter to use with the Gemma model"
            )
        
        # Add explanatory notes
        st.info("""
        ### What You're Downloading
        
        This is a **small adapter file** (typically just a few MB), NOT the full Gemma model.
        
        **To use your custom AI:**
        1. Download this adapter
        2. Get the base Gemma model from Hugging Face
        3. Load both together using the included example code
        
        The adapter contains all the customizations from your training data!
        """)
        
        # Display the location of the adapter
        st.markdown(f"**Adapter saved to:** `{output_dir}`")
        
        # Clean up the temporary file
        os.unlink(temp_zip_path)
        
    except Exception as e:
        st.error(f"Error preparing download: {str(e)}")
    
    # If we have training history, plot the loss curve
    if training_history and len(training_history) > 0:
        st.markdown("### Training Loss")
        st.caption("**Lower values indicate better learning.** The model improves as training progresses.")
        
        try:
            import pandas as pd
            import plotly.express as px
            
            # Create dataframe for plotting
            df = pd.DataFrame({
                "Step": list(range(1, len(training_history) + 1)),
                "Loss": training_history
            })
            
            # Create interactive plot
            fig = px.line(
                df, x="Step", y="Loss", 
                title="Training Loss Over Time",
                labels={"Loss": "Training Loss (lower is better)"}
            )
            fig.update_layout(
                title_font_size=20,
                xaxis_title_font_size=16,
                yaxis_title_font_size=16
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add some interpretation
            if len(training_history) > 1:
                initial_loss = training_history[0]
                final_loss = training_history[-1]
                improvement = (initial_loss - final_loss) / initial_loss * 100
                
                if improvement > 0:
                    st.success(f"Loss improved by {improvement:.1f}% during training! (from {initial_loss:.4f} to {final_loss:.4f})")
                else:
                    st.warning("The model didn't show significant improvement. Consider training for longer or with a different dataset.")
                
        except Exception as e:
            st.error(f"Error plotting training history: {str(e)}")

def show_demo_data():
    """
    Display and provide download for the demo dataset.
    """
    # Get the demo dataset
    demo_dataset, csv_string = get_demo_dataset()
    
    # Improved demo data display with clearer column explanations
    demo_df = pd.DataFrame({
        "prompt": [
            "Explain what machine learning is to a 10-year-old.",
            "Write a short poem about the night sky.",
            "What are three tips for staying healthy?",
            "Summarize what photosynthesis is.",
            "How does recycling help the environment?"
        ],
        "completion": [
            "Machine learning is like teaching a computer to learn from examples...",
            "Silent diamonds in the night,\nWhispering stories of ancient light...",
            "1. Drink plenty of water every day to stay hydrated.\n2. Get at least 30 minutes...",
            "Photosynthesis is how plants make their own food. They use sunlight...",
            "Recycling helps the environment by reducing waste in landfills..."
        ]
    })
    
    # Display with clearer column headers and explanations
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Inputs (what you ask)**")
    with col2:
        st.markdown("**Outputs (how you want the AI to respond)**")

    # Show the dataframe with improved styling
    st.dataframe(
        demo_df,
        column_config={
            "prompt": "Your Prompts",
            "completion": "AI's Response"
        },
        height=300
    )
    
    # Add download button for the full CSV with better explanation
    st.download_button(
        label="📥 Download This Sample Dataset (.csv)",
        data=csv_string,
        file_name="gemma_sample_data.csv",
        mime="text/csv",
        help="Download this sample dataset to see the expected format or use it for a test run."
    )
    
    st.info("""
    **Your dataset should follow this format:**
    
    - **prompt column:** Questions or instructions you want to give the AI
    - **completion column:** How you want the AI to respond in each case
    
    For best results, include at least 10-50 diverse examples that represent how you want the AI to respond. 
    The more examples, the better the AI will learn your style!
    """)
    
    return demo_dataset

def display_system_check(device="cpu", device_name="Unknown", device_info="CPU only"):
    """
    Display information about the system and check for compatibility.
    
    Args:
        device: The device being used ('cuda', 'mps', or 'cpu')
        device_name: The name of the device
        device_info: Additional device information
    """
    system_col1, system_col2 = st.columns(2)
    
    with system_col1:
        st.markdown("### Hardware")
        
        # GPU information with clearer explanation of requirements
        st.markdown(f"**Device:** {device_name}")
        st.markdown(f"**Type:** {device_info}")
        
        # Make GPU requirements clearer
        if device == "cuda":
            st.success("✅ **NVIDIA GPU Detected:** Your system can run training!")
            # Check VRAM and give additional guidance
            try:
                import torch
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if vram_gb < 8:
                    st.warning(f"⚠️ Limited VRAM: {vram_gb:.1f} GB. Enable 4-bit mode and use Gemma 2B.")
                elif vram_gb < 12:
                    st.info(f"ℹ️ VRAM: {vram_gb:.1f} GB. Enable 4-bit mode for Gemma 7B.")
                else:
                    st.success(f"✅ VRAM: {vram_gb:.1f} GB. Good for Gemma 7B!")
            except:
                st.info("Could not determine VRAM amount. Monitor memory usage during training.")
        else:
            if device == "mps":
                st.warning("⚠️ **Apple Silicon Detected:** Limited testing mode only. Full training requires NVIDIA GPU.")
            else:
                st.error("❌ **No GPU Detected:** Training requires an NVIDIA GPU with CUDA support.")
    
    with system_col2:
        st.markdown("### Software")
        
        # Python version check
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info.major == 3 and sys.version_info.minor >= 9:
            st.success(f"✅ **Python {python_version}:** Compatible")
        else:
            st.error(f"❌ **Python {python_version}:** Python 3.9+ required")
        
        # PyTorch check with clearer explanation
        import torch
        torch_version = torch.__version__
        st.markdown(f"**PyTorch:** {torch_version}")
        
        # CUDA check with clearer explanation
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            st.success(f"✅ **CUDA {cuda_version}:** Ready for training")
        else:
            if device == "mps":
                st.warning("⚠️ **MPS:** Apple Silicon GPU (limited support)")
            else:
                st.error("❌ **CUDA:** Not available (required for training)")
    
    # Clear recommendations
    st.markdown("### Recommended Setup")
    
    if device != "cuda":
        st.error("""
        For proper training, you need:
        - NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
        - CUDA toolkit installed
        - PyTorch with CUDA support
        
        **Alternative:** Use a cloud provider like Google Colab, vast.ai, or runpod.io
        """)
    elif device == "cuda":
        import torch
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if vram_gb < 8:
            st.warning(f"""
            Your GPU has limited memory ({vram_gb:.1f} GB):
            - Use the 2B model instead of 7B
            - Enable 4-bit quantization
            - Keep batch size at 1
            - Limit dataset size to 100-200 examples
            """)
        elif vram_gb < 16:
            st.info(f"""
            Your GPU has {vram_gb:.1f} GB memory:
            - Use 4-bit quantization for the 7B model
            - Keep batch size between 1-4
            - Dataset can be up to 1000 examples
            """)
        else:
            st.success(f"""
            Your GPU has good memory ({vram_gb:.1f} GB):
            - Can use all models with 4-bit quantization
            - Can use larger batch sizes (4-8)
            - Can handle larger datasets (1000+ examples)
            """)
