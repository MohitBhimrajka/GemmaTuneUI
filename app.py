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

# Main application title and welcome section
st.title("✨ GemmaTuneUI: Easily Customize Your Own AI ✨")

# Welcome Section with explanation
with st.container():
    st.markdown("""
    ## Welcome to GemmaTuneUI!
    
    **What is this tool?** This tool helps you fine-tune Google's Gemma AI models to follow your instructions 
    or match your writing style better.
    
    **Why use it?** Create a personalized AI assistant, make Gemma better at specific tasks (like summarizing 
    your notes, writing emails in your tone), or just experiment with AI customization!
    
    **What you need:**
    1. A dataset (CSV or JSONL) with examples of what you want the AI to learn (see sample below!)
    2. A computer with a reasonably powerful NVIDIA GPU (RTX 3060 or better recommended) with CUDA installed
    3. Patience - training can take time depending on your dataset size and GPU power
    
    **What you get:** A small 'adapter' file. This file modifies the base Gemma model to act according to your data.
    """)

    # Show sample dataset
    with st.expander("▶️ See a sample dataset", expanded=True):
        demo_dataset = show_demo_data()

# System compatibility check section
with st.expander("▶️ System Compatibility Check", expanded=False):
    display_system_check()

# Workflow steps
st.markdown("---")

# Step 1: Configure model & training
st.header("Step 1: Configure Your Model & Training")
# Sidebar rendering
config = render_sidebar(get_default_config(), get_parameter_explanations())
st.caption("Settings are on the left sidebar. Hover over (?) icons for help!")

# Step 2: Upload data
st.header("Step 2: Upload Your Data")
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or JSONL)", 
    type=["csv", "json", "jsonl"],
    help="Your file should have a 'prompt' column for inputs and a 'completion' column for desired outputs."
)

# Add option to use demo data
use_demo = st.checkbox("Or use the sample dataset for a test run", value=False)

# Step 3: Preview & format data
st.header("Step 3: Preview & Format Data")

dataset = None
if uploaded_file is not None:
    # If a file was uploaded, use it
    dataset = load_and_format_dataset(uploaded_file)
    if dataset is not None:
        st.session_state.formatted_dataset = dataset
        st.success(f"✅ Dataset loaded successfully with {len(dataset)} examples!")
elif use_demo:
    # If using demo data, get it from the demo function
    st.info("Using the sample dataset for this run.")
    dataset = demo_dataset
    st.session_state.formatted_dataset = dataset
    st.success(f"✅ Sample dataset loaded with {len(dataset)} examples!")
else:
    # No data yet
    st.info("Please upload a file or use the sample dataset to continue.")

# Step 4: Start fine-tuning
st.header("Step 4: Start Fine-Tuning!")

if st.session_state.formatted_dataset is not None:
    # Display warning about training time
    st.warning("""
    **Training can take time!** Depending on your GPU and dataset size, this could take anywhere 
    from a few minutes to several hours. During training, the application will be unresponsive.
    
    Make sure your computer won't go to sleep during training.
    """)
    
    # Start training button
    start_training = st.button("🚀 Start Fine-Tuning", use_container_width=True)
    
    if start_training:
        try:
            # Create progress placeholders
            progress_bar, status_text, log_container = display_training_progress()
            
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
            with st.spinner("Training in progress..."):
                output_dir = trainer.train(
                    st.session_state.formatted_dataset,
                    progress_bar,
                    status_text,
                    log_container
                )
            
            # Update session state to indicate training completion
            st.session_state.training_complete = True
            
            # Add success message after training
            st.success("🎉 Fine-tuning complete!")
            
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            st.code(traceback.format_exc())
            st.session_state.training_complete = False

# Step 5: Get your custom adapter
st.header("Step 5: Get Your Custom AI Adapter")

if st.session_state.training_complete:
    # Display results and download option
    display_results(st.session_state.output_dir, st.session_state.training_history)
else:
    st.info("Your custom AI adapter will appear here after fine-tuning is complete.")

# Footer with information
st.markdown("---")
st.markdown("""
**GemmaTuneUI** is an open-source tool for customizing Google's Gemma models. 
It uses Parameter-Efficient Fine-Tuning (PEFT) with QLoRA to make fine-tuning 
accessible on consumer hardware.

Made with ❤️ using Streamlit, Transformers, and PEFT.
""")

# Add a button to clear the session state
if st.button("Start Over", help="Clear all data and start a new fine-tuning session"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()
