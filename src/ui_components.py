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
    
    # Model section
    st.sidebar.header("Model")
    config["model_name"] = st.sidebar.selectbox(
        "Gemma Model",
        options=default_config["model_options"],
        index=default_config["model_options"].index(default_config["model_name"]),
        help=explanations["model_name"]
    )
    
    # Quantization section
    st.sidebar.header("Memory Optimization")
    config["use_4bit"] = st.sidebar.checkbox(
        "Use 4-bit Quantization (QLoRA)",
        value=default_config["use_4bit"],
        help=explanations["use_4bit"]
    )
    
    # Training parameters section
    st.sidebar.header("Training Parameters")
    
    config["num_train_epochs"] = st.sidebar.number_input(
        "Number of Epochs",
        min_value=1,
        max_value=10,
        value=default_config["num_train_epochs"],
        help=explanations["num_train_epochs"]
    )
    
    config["learning_rate"] = st.sidebar.number_input(
        "Learning Rate",
        min_value=1e-6,
        max_value=1e-3,
        value=default_config["learning_rate"],
        format="%.1e",
        help=explanations["learning_rate"]
    )
    
    config["per_device_train_batch_size"] = st.sidebar.number_input(
        "Batch Size",
        min_value=1,
        max_value=8,
        value=default_config["per_device_train_batch_size"],
        help=explanations["per_device_train_batch_size"]
    )
    
    config["gradient_accumulation_steps"] = st.sidebar.number_input(
        "Gradient Accumulation Steps",
        min_value=1,
        max_value=16,
        value=default_config["gradient_accumulation_steps"],
        help=explanations["gradient_accumulation_steps"]
    )
    
    # LoRA parameters section - use an expander to keep UI clean
    st.sidebar.header("LoRA Parameters")
    with st.sidebar.expander("Advanced LoRA Settings"):
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
    
    # Add a GPU memory indicator
    if config["model_name"].startswith("google/gemma-2b"):
        gpu_mem_needed = "~8GB VRAM" if config["use_4bit"] else "~14GB VRAM"
    else:  # 7B model
        gpu_mem_needed = "~14GB VRAM" if config["use_4bit"] else "~28GB VRAM"
    
    st.sidebar.info(f"Estimated GPU Memory Needed: {gpu_mem_needed}")
    
    return config

def display_training_progress(title="Training Progress"):
    """
    Create and display placeholders for training progress.
    
    Args:
        title: Title for the training progress section
        
    Returns:
        tuple: (progress_bar, status_text, log_container)
    """
    st.subheader(title)
    progress_bar = st.progress(0)
    status_container = st.empty()
    status_text = status_container.text("Initializing...")
    log_container = st.empty()
    
    return progress_bar, status_text, log_container

def display_results(output_dir, training_loss=None):
    """
    Display training results and provide download functionality.
    
    Args:
        output_dir: Directory containing saved model
        training_loss: List of loss values from training (if available)
    """
    st.subheader("Training Results")
    
    # Display loss plot if available
    if training_loss and len(training_loss) > 0:
        st.write("### Training Loss")
        
        # Create a simple loss plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=training_loss,
            mode='lines',
            name='Loss',
            line=dict(color='rgba(50, 168, 82, 1)')
        ))
        
        fig.update_layout(
            title="Training Progress (Lower Loss is Better)",
            xaxis_title="Steps",
            yaxis_title="Loss",
            height=400,
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("""
        **Understanding Loss**: The loss value shows how well the model is learning. 
        Lower values mean better performance. The line should generally trend downward as training progresses.
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
            
            # Create download button
            st.download_button(
                label="📥 Download Fine-Tuned AI Adapter (.zip)",
                data=zip_data,
                file_name=zip_name,
                mime="application/zip",
                help="Download the adapter that contains your fine-tuned model adjustments."
            )
            
            st.info("""
            **What you're downloading**: This file contains the *changes* made to the base model during training.
            It's much smaller than the full model (just a few MB instead of several GB).
            
            To use it later, you'll need to:
            1. Load the original Gemma model
            2. Apply this adapter to it using the PEFT library
            3. The README.md file in the zip has detailed instructions!
            """)
    else:
        st.error(f"Output directory {output_dir} not found.")

def show_demo_data():
    """
    Display and provide download for the demo dataset.
    """
    # Get the demo dataset
    demo_dataset, csv_string = get_demo_dataset()
    
    # Create a DataFrame from the dataset for display
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
    
    st.write("### Sample Dataset Format")
    st.dataframe(demo_df)
    
    # Add download button for the full CSV
    st.download_button(
        label="📥 Download Sample Dataset (.csv)",
        data=csv_string,
        file_name="gemma_sample_data.csv",
        mime="text/csv",
        help="Download this sample dataset to see the expected format or use it for a test run."
    )
    
    st.info("""
    This sample dataset shows the expected format for fine-tuning. 
    Your dataset should have:
    - A **prompt** column with user inputs or questions
    - A **completion** column with the desired AI responses
    
    For best results, include at least 10-50 examples that represent the style and type of responses you want the model to learn.
    """)
    
    return demo_dataset

def display_system_check():
    """
    Display system compatibility information.
    """
    st.subheader("System Compatibility Check")
    
    import torch
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
        
        st.success(f"✅ CUDA is available with {gpu_count} GPU(s)")
        st.write(f"GPU: {gpu_name}")
        st.write(f"GPU Memory: {gpu_memory:.2f} GB")
        
        # Check memory requirements
        if gpu_memory < 8:
            st.warning("⚠️ Your GPU has less than 8GB memory. This may limit the models you can fine-tune.")
            st.write("Recommendation: Use the 2B model with 4-bit quantization enabled.")
        elif gpu_memory < 16:
            st.write("✅ Your GPU should work well with Gemma 2B models.")
            st.write("ℹ️ For Gemma 7B models, 4-bit quantization is strongly recommended.")
        else:
            st.write("✅ Your GPU should work well with all Gemma models.")
    else:
        st.error("❌ CUDA is not available. GPU acceleration is required for fine-tuning.")
        st.write("""
        This tool requires an NVIDIA GPU with CUDA support. Without it, model fine-tuning will be extremely slow or may fail.
        
        If you have an NVIDIA GPU:
        1. Make sure you have the latest NVIDIA drivers installed
        2. Install CUDA toolkit and cuDNN
        3. Install PyTorch with CUDA support
        """)
    
    # Check for bitsandbytes
    try:
        import bitsandbytes as bnb
        st.write("✅ bitsandbytes is installed correctly for 4-bit quantization")
    except ImportError:
        st.warning("⚠️ bitsandbytes library not found. Quantization features will not work.")
    except Exception as e:
        st.warning(f"⚠️ bitsandbytes issue: {str(e)}")
    
    # Show available disk space
    import shutil
    disk_usage = shutil.disk_usage("/")
    disk_free_gb = disk_usage.free / (1024**3)
    
    if disk_free_gb < 5:
        st.warning(f"⚠️ Low disk space: {disk_free_gb:.1f} GB free. At least 5GB recommended.")
    else:
        st.write(f"✅ Available disk space: {disk_free_gb:.1f} GB")
