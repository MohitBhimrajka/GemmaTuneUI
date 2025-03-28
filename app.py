"""
GemmaTuneUI: A user-friendly web application for fine-tuning Google's Gemma models.
"""

import streamlit as st
import os
import pandas as pd
import torch
import traceback
from datetime import datetime

# Import from our modules
from src.utils import get_default_config, get_parameter_explanations
from src.ui_components import (
    render_sidebar, 
    display_training_progress, 
    display_results, 
    show_demo_data,
    display_system_check
)
from src.data_handler import load_and_format_dataset
from src.trainer import GemmaTrainer

# Configure the page
st.set_page_config(
    page_title="GemmaTuneUI - Easy Fine-Tuning", 
    page_icon="✨",
    layout="wide"
)

# Define session state initialization
if "formatted_dataset" not in st.session_state:
    st.session_state.formatted_dataset = None
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False
if "training_history" not in st.session_state:
    st.session_state.training_history = []
if "output_dir" not in st.session_state:
    st.session_state.output_dir = "./lora-gemma-output"

# Check if we're in Mac testing mode (set by run.sh)
MAC_TESTING_MODE = os.environ.get("MAC_TESTING_MODE", "0") == "1"
IS_APPLE_SILICON = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False

# Detect available devices
if torch.cuda.is_available():
    DEVICE = "cuda"
    DEVICE_NAME = torch.cuda.get_device_name(0)
    DEVICE_INFO = f"NVIDIA GPU: {DEVICE_NAME}"
elif IS_APPLE_SILICON:
    DEVICE = "mps"
    DEVICE_NAME = "Apple Silicon"
    DEVICE_INFO = f"Apple Silicon (M1/M2/M3) - Limited functionality"
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"
    DEVICE_INFO = "CPU only (training will be extremely slow or may fail)"

# Main application title and welcome section with improved styling
st.title("✨ GemmaTuneUI: Easily Customize Your Own AI ✨")

# Welcome Section with clearer explanation and visual separation
with st.container():
    welcome_col1, welcome_col2 = st.columns([2, 1])
    
    with welcome_col1:
        st.markdown("""
        ## Welcome to GemmaTuneUI!
        
        **What is this tool?** 
        This tool helps you fine-tune Google's Gemma AI models to follow your instructions or match your writing style better.
        
        **Why use it?** 
        Create a personalized AI assistant, make Gemma better at specific tasks (like summarizing notes or writing emails in your tone), or simply experiment with AI customization!
        
        **What you'll get:** 
        A small 'adapter' file (just a few MB) that modifies the base Gemma model to act according to your examples.
        """)
    
    with welcome_col2:
        st.info("""
        ### You'll need:
        
        ✅ A dataset with examples (or use our sample!)
        
        ✅ A computer with an NVIDIA GPU (RTX 3060+ recommended)
        
        ✅ CUDA installed 
        """)

# Show sample dataset prominently in the welcome section
st.markdown("### 📋 Sample Dataset Format")
st.caption("Here's what your data should look like. You can use this sample to test the app!")
demo_dataset = show_demo_data()

# System compatibility check section - expanded by default for first-time users
with st.expander("🖥️ Check Your System Compatibility", expanded=True):
    display_system_check()

# Clear workflow steps with visual separators
st.markdown("---")
st.markdown("## Follow these 5 simple steps:")

# Step 1: Configure model & training - clearer guidance
st.header("Step 1: Configure Your Model & Training")
st.caption("👈 Settings are on the left sidebar. Hover over any (?) icon for simple explanations of each option.")
# Sidebar rendering
config = render_sidebar(get_default_config(), get_parameter_explanations())

# Step 2: Upload data - improved upload UI with clearer instructions
st.header("Step 2: Upload Your Dataset")
st.markdown("""
Choose one of these options:
1. **Upload your own dataset** (CSV or JSONL file with examples of what you want the AI to learn)
2. **Use our sample dataset** to try things out
""")

uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or JSONL)", 
    type=["csv", "json", "jsonl"],
    help="Your file should have columns like 'prompt'/'completion' or 'instruction'/'response' with your examples."
)

# Add option to use demo data
use_demo = st.checkbox("👉 Or use the sample dataset shown above for a test run", value=False)

# Step 3: Preview & format data - improved feedback
st.header("Step 3: Preview & Format Your Data")

dataset = None
if uploaded_file is not None:
    # If a file was uploaded, use it
    with st.spinner("Loading and formatting your dataset..."):
        dataset = load_and_format_dataset(uploaded_file)
    if dataset is not None:
        st.session_state.formatted_dataset = dataset
        st.success(f"✅ Dataset loaded successfully with {len(dataset)} examples!")
elif use_demo:
    # If using demo data, get it from the demo function
    with st.spinner("Preparing sample dataset..."):
        st.info("Using the sample dataset for this run.")
        dataset = demo_dataset
        st.session_state.formatted_dataset = dataset
        st.success(f"✅ Sample dataset loaded with {len(dataset)} examples!")
else:
    # No data yet - clearer call to action
    st.info("👆 Please upload a file or use the sample dataset to continue to Step 4.")

# Step 4: Start fine-tuning - improved warning and button
st.header("Step 4: Start Fine-Tuning!")

if st.session_state.formatted_dataset is not None:
    # Display warning about training time
    st.warning("""
    ⏱️ **Training can take time!** Depending on your GPU and dataset size, this could take anywhere 
    from a few minutes to several hours. The app will show progress but may appear frozen at times.
    
    💡 **Tip:** Make sure your computer won't go to sleep during training.
    """)
    
    # Start training button - more prominent
    start_training = st.button("🚀 Start Fine-Tuning", use_container_width=True, type="primary")
    
    if start_training:
        # Check if we're in Mac testing mode
        if MAC_TESTING_MODE and DEVICE != "cuda":
            st.warning("""
            ### ⚠️ Mac Testing Mode Warning
            
            You're attempting to train on a Mac without NVIDIA GPU acceleration.
            This will likely fail or be extremely slow.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                continue_anyway = st.button("Yes, Continue Anyway", type="primary")
                if not continue_anyway:
                    st.stop()
            with col2:
                if st.button("Cancel", type="secondary"):
                    st.stop()
                    
            if not continue_anyway:
                st.stop()
        
        try:
            # Progress indicators with detailed status updates
            st.subheader("Training Progress")
            st.markdown("Your AI is learning from your examples. This may take some time...")
            
            # Create progress placeholders
            progress_bar, status_text, log_container = display_training_progress(title=None)
            
            # Set up output directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(config["output_dir"], f"gemma_lora_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Update config with new output directory
            config["output_dir"] = output_dir
            st.session_state.output_dir = output_dir
            
            # Initialize trainer
            trainer = GemmaTrainer(config)
            
            # Start training with Streamlit progress reporting
            with st.spinner("AI training in progress..."):
                output_dir = trainer.train(
                    st.session_state.formatted_dataset,
                    progress_bar,
                    status_text,
                    log_container
                )
            
            # Update session state to indicate training completion
            st.session_state.training_complete = True
            
            # Clear success message after training
            st.balloons()
            st.success("🎉 Fine-tuning complete! Your personalized AI adapter is ready below.")
            
        except Exception as e:
            # More helpful error handling with suggestions based on error type
            error_msg = str(e)
            st.error(f"Training error: {error_msg}")
            
            # Provide specific advice based on common errors
            if "CUDA out of memory" in error_msg:
                st.error("""
                💡 **Memory Issue Detected**: Your GPU ran out of memory. Try:
                1. Enabling 4-bit quantization (QLoRA) in the sidebar
                2. Reducing batch size to 1 or 2
                3. Using the smaller Gemma 2B model instead of 7B
                """)
            elif "No CUDA GPUs are available" in error_msg:
                st.error("""
                💡 **GPU Issue Detected**: CUDA GPU not found. This app requires an NVIDIA GPU with CUDA support.
                Please check that your GPU drivers are installed correctly.
                """)
            elif "No such file or directory" in error_msg:
                st.error("""
                💡 **File Error**: A required file could not be found. This may be due to a download error.
                Try restarting the application.
                """)
            
            st.code(traceback.format_exc())
            st.session_state.training_complete = False

# Step 5: Get your custom adapter - improved explanation
st.header("Step 5: Get Your Custom AI Adapter")

if st.session_state.training_complete:
    # Display results and download option
    display_results(st.session_state.output_dir, st.session_state.training_history)
else:
    # Clearer waiting message
    st.info("💤 Your personalized AI adapter will appear here after training is complete.")

# Footer with information
st.markdown("---")
st.markdown("""
**GemmaTuneUI** is an open-source tool for customizing Google's Gemma models. 
It uses Parameter-Efficient Fine-Tuning (PEFT) with QLoRA to make fine-tuning 
accessible on consumer hardware.

Made with ❤️ using Streamlit, Transformers, and PEFT.
""")

# Add a button to clear the session state - better labeled
if st.button("🔄 Start Over", help="Clear all data and start a new fine-tuning session"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()
