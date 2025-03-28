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

def display_results(output_dir, training_loss=None):
    """
    Display training results and provide download functionality.
    
    Args:
        output_dir: Directory containing saved model
        training_loss: List of loss values from training (if available)
    """
    st.subheader("🏆 Your Personalized AI Adapter is Ready!")
    
    # Display loss plot if available with improved explanation
    if training_loss and len(training_loss) > 0:
        st.write("### 📉 Training Progress")
        
        # Create a simpler, more intuitive loss plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=training_loss,
            mode='lines',
            name='Loss',
            line=dict(color='rgba(76, 175, 80, 1)', width=3)
        ))
        
        fig.update_layout(
            title="Training Progress (Lower Numbers = Better Learning)",
            xaxis_title="Training Steps",
            yaxis_title="Loss Value",
            height=400,
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("""
        **What is this chart?** This shows how well your AI was learning. 
        The line should generally go down over time, meaning the AI is getting better at matching your examples.
        """)
    
    # Check if the output directory exists
    if os.path.exists(output_dir):
        from src.trainer import GemmaTrainer
        
        # Create a trainer instance to generate the zip file
        # We need config for this, so we'll use a dummy config
        dummy_config = {"model_name": "google/gemma-2b", "output_dir": output_dir}
        trainer = GemmaTrainer(dummy_config)
        
        # Create zip file
        zip_path, zip_name = trainer.create_adapter_zip()
        
        if zip_path and os.path.exists(zip_path):
            # Read the zip file
            with open(zip_path, "rb") as f:
                zip_data = f.read()
            
            # Create more prominent download section
            st.markdown("### 📥 Download Your AI Adapter")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.download_button(
                    label="📥 Download Fine-Tuned AI Adapter (.zip)",
                    data=zip_data,
                    file_name=zip_name,
                    mime="application/zip",
                    help="Download the adapter that contains your fine-tuned model adjustments.",
                    use_container_width=True
                )
            
            # Clearer explanation of what they're downloading
            st.info("""
            **What you're downloading:** This zip file contains your AI adapter - it's the *changes* made to the 
            base Gemma model during training. It's much smaller than the full model (just a few MB instead of several GB).
            
            **To use it later:**
            1. You'll still need the original Gemma model
            2. This adapter attaches to it to create your personalized version
            3. The README.md file in the zip has step-by-step instructions!
            """)
            
            # Show example usage code
            with st.expander("See example code for using your adapter"):
                st.code("""
# Python code to use your adapter with the base Gemma model
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model and tokenizer
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")  # Use the same model you selected
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

# Load your adapter (after unzipping it)
model = PeftModel.from_pretrained(model, "./your_adapter_directory")

# Use your personalized model
response = model.generate(
    **tokenizer("Write me a short poem about the ocean", return_tensors="pt"),
    max_length=200
)
print(tokenizer.decode(response[0], skip_special_tokens=True))
                """, language="python")
    else:
        st.error(f"Output directory {output_dir} not found.")

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

def display_system_check():
    """
    Display system compatibility information with clearer explanations and visual indicators.
    """
    import torch
    
    # Check for CUDA availability with more user-friendly output
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
        
        st.success(f"✅ **GPU Detected:** {gpu_name}")
        st.markdown(f"**Available VRAM:** {gpu_memory:.1f} GB")
        st.markdown(f"**CUDA Version:** {torch.version.cuda}")
        
        # Check memory requirements with clearer recommendations
        if gpu_memory < 8:
            st.warning("""
            ⚠️ **Limited GPU Memory:** Your GPU has less than 8GB memory, which may be insufficient.
            
            **Recommendations:**
            - Use the Gemma 2B model (not 7B)
            - Keep 4-bit quantization (QLoRA) enabled
            - Set batch size to 1
            - You may still encounter memory errors
            """)
        elif gpu_memory < 16:
            st.success("""
            ✅ **Suitable GPU Memory:** Your GPU should work well with Gemma 2B models.
            
            **Recommendations:**
            - Keep 4-bit quantization (QLoRA) enabled for best results
            - For Gemma 7B models, use a batch size of 1
            """)
        else:
            st.success("""
            ✅ **Excellent GPU Memory:** Your GPU should work well with all Gemma models.
            
            You can use either 2B or 7B models with good performance.
            """)
    else:
        st.error("""
        ❌ **No CUDA GPU Detected:** This application requires an NVIDIA GPU with CUDA support.
        
        **Without a compatible GPU:**
        - Fine-tuning will fail or be extremely slow
        - You may get CUDA-related errors
        
        **What you need:**
        - An NVIDIA GPU (RTX 3060 or better recommended)
        - Properly installed NVIDIA drivers
        - CUDA toolkit installed (version 11.8+ recommended)
        """)
    
    # Check for bitsandbytes with clearer explanation
    try:
        import bitsandbytes as bnb
        st.success("✅ **Quantization library (bitsandbytes) installed correctly**")
        st.caption("This enables 4-bit quantization (QLoRA) for memory-efficient training")
    except ImportError:
        st.warning("""
        ⚠️ **bitsandbytes library not found**
        
        This is required for 4-bit quantization (QLoRA). Without it, memory usage will be much higher.
        Try reinstalling requirements: `pip install -r requirements.txt`
        """)
    except Exception as e:
        st.warning(f"⚠️ **bitsandbytes issue:** {str(e)}")
    
    # Show available disk space with clearer formatting
    import shutil
    disk_usage = shutil.disk_usage("/")
    disk_free_gb = disk_usage.free / (1024**3)
    
    if disk_free_gb < 5:
        st.warning(f"⚠️ **Low disk space:** {disk_free_gb:.1f} GB free. At least 5GB recommended.")
    else:
        st.success(f"✅ **Disk space available:** {disk_free_gb:.1f} GB free")
        
    # Show a summary of system readiness
    st.markdown("---")
    if torch.cuda.is_available() and disk_free_gb >= 5:
        st.success("✅ **Your system is ready for fine-tuning!**")
    else:
        st.warning("⚠️ **Your system may not be fully compatible** - check the warnings above.")
