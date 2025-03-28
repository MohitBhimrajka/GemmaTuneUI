"""
GemmaTuneUI: A user-friendly web application for fine-tuning Google's Gemma models.
"""

import streamlit as st
import os
import json
import threading
import pandas as pd
import torch
import traceback
import time
from datetime import datetime
from pathlib import Path

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

# Handle Mac-specific compatibility issues
try:
    import bitsandbytes as bnb
except ImportError:
    # bitsandbytes not installed, handled in trainer
    pass
except Exception as e:
    # bitsandbytes compatibility issue, will use fallback
    # Don't warn here - we'll handle compatibility at runtime
    pass

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
if "training_status" not in st.session_state:
    st.session_state.training_status = "idle"  # idle, running, completed, error
if "training_error" not in st.session_state:
    st.session_state.training_error = None
if "demo_dataset" not in st.session_state:
    st.session_state.demo_dataset = None

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

# Training function for background thread
def train_in_background(dataset, config, progress_bar, status_text, log_container):
    try:
        # Mac compatibility check
        if IS_APPLE_SILICON or DEVICE == "cpu":
            status_text.text("⚠️ Running in Mac compatibility mode...")
            # Force 8-bit and 4-bit quantization off in Mac environments
            if config.get("use_8bit_quantization", False):
                status_text.text("⚠️ 8-bit quantization disabled (not compatible with Mac)")
                config["use_8bit_quantization"] = False
                time.sleep(1)  # Let user see the message
            
            if config.get("use_4bit_quantization", False):
                status_text.text("⚠️ 4-bit quantization automatically disabled (Mac compatibility)")
                config["use_4bit_quantization"] = False 
                time.sleep(1)
        
        # Log starting configuration
        status_text.text("Starting trainer with selected configuration...")
        log_container.text(f"Model: {config.get('model_name')}")
        log_container.text(f"Device: {DEVICE} ({DEVICE_NAME})")
        log_container.text(f"Examples: {len(dataset)}")
        
        try:
            # Initialize trainer
            trainer = GemmaTrainer(config)
            
            # Start training with Streamlit progress reporting
            output_dir = trainer.train(
                dataset,
                progress_bar,
                status_text,
                log_container
            )
            
            # Update session state to indicate training completion
            st.session_state.training_complete = True
            st.session_state.training_status = "completed"
            st.session_state.output_dir = output_dir
        except AttributeError as e:
            if any(err in str(e) for err in ["cadam32bit", "bnb", "bitsandbytes", "quantiz"]):
                raise RuntimeError(f"Mac compatibility issue: {str(e)}. Please disable quantization options.")
            raise  # Re-raise other attribute errors
        
    except torch.cuda.OutOfMemoryError as e:
        # Handle CUDA OOM errors specifically
        st.session_state.training_error = "GPU ran out of memory. Try enabling 4-bit quantization, reducing batch size, or using a smaller model."
        st.session_state.training_status = "error"
        st.session_state.training_complete = False
        
    except ImportError as e:
        # Handle missing dependencies
        st.session_state.training_error = f"Missing required library: {str(e)}. Please check your installation."
        st.session_state.training_status = "error"
        st.session_state.training_complete = False
    
    except Exception as e:
        # Generic error handler
        st.session_state.training_error = str(e)
        st.session_state.training_status = "error"
        st.session_state.training_complete = False

# Main app function
def main():
    # Main application title and welcome section with improved styling
    st.title("✨ GemmaTuneUI: Easily Customize Your Own AI ✨")
    
    # If in Mac testing mode, show a special banner
    if MAC_TESTING_MODE:
        st.warning("""
        ### Mac Testing Mode Enabled
        
        You're running GemmaTuneUI on a Mac, which has limited GPU support for model training.
        
        **What will work:**
        - Exploring the UI and workflow
        - Loading and validating datasets
        - Configuring training parameters (except 8-bit quantization)
        
        **What may not work:**
        - 8-bit quantization (automatically disabled)
        - Actual model training (may fail or be extremely slow)
        - GPU-accelerated inference
        
        For production use, we recommend using a system with an NVIDIA GPU.
        """)

    # Make the content more accessible and welcoming
    with st.container():
        st.markdown("""
        **Create your own customized AI assistant in minutes!** 
        
        GemmaTuneUI lets you train a personalized version of Google's Gemma AI model on your own examples, 
        even without technical expertise or expensive hardware.
        """)

    # Clear demo data section - shows early in the workflow so users see it first
    with st.expander("👀 See a sample dataset to get started", expanded=True):
        st.markdown("Here's what a good training dataset might look like. You can use this sample to test the app!")
    
    # Load demo dataset early
    with st.spinner("Loading demo dataset..."):
        demo_dataset = show_demo_data()
        st.session_state.demo_dataset = demo_dataset

    # System compatibility check section - expanded by default for first-time users
    with st.expander("🖥️ Check Your System Compatibility", expanded=True):
        display_system_check(device=DEVICE, device_name=DEVICE_NAME, device_info=DEVICE_INFO)
        
        # Add stronger warning for non-CUDA devices
        if DEVICE != "cuda":
            st.error("""
            ### ⚠️ WARNING: Training Requires NVIDIA GPU ⚠️
            
            Fine-tuning AI models is extremely computationally intensive and requires specialized hardware.
            
            **Your system does not have an NVIDIA GPU with CUDA support, which is required for proper training.**
            
            Options:
            1. Continue in testing mode to explore the interface and workflow
            2. Use Google Colab, vast.ai, or runpod.io (cloud services with GPU access)
            3. Try on a different computer with an NVIDIA GPU
            
            **Note:** Attempting to train on this system will be extremely slow (hours/days) or will fail completely.
            """)

    # Clear workflow steps with visual separators
    st.markdown("---")
    st.markdown("## Follow these 5 simple steps:")

    # Step 1: Configure model & training - clearer guidance
    st.header("Step 1: Configure Your Model & Training")
    st.caption("👈 Settings are on the left sidebar. Hover over any (?) icon for simple explanations of each option.")
    
    # Get default config
    config = get_default_config()
    
    # Disable quantization features by default on Mac systems
    if MAC_TESTING_MODE or IS_APPLE_SILICON:
        config["use_8bit_quantization"] = False
        config["use_4bit_quantization"] = False
        # Use more Mac-friendly settings
        config["batch_size"] = 1  # Smaller batch size
        config["gradient_accumulation_steps"] = 4  # Use gradient accumulation instead
        if "7b" in config["model_name"].lower():
            st.warning("⚠️ On Mac systems, the smaller 2B model is recommended for better compatibility")
    
    # Sidebar rendering
    config = render_sidebar(config, get_parameter_explanations())

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
            # Use cached demo dataset
            dataset = st.session_state.demo_dataset
            if dataset is None:
                dataset = show_demo_data()
                st.session_state.demo_dataset = dataset
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
        
        # Training status display
        if st.session_state.training_status == "running":
            st.info("### 🔄 Training is currently in progress...")
            
            # Progress indicators with detailed status updates
            st.subheader("Training Progress")
            st.markdown("Your AI is learning from your examples. This may take some time...")
            
            # Create progress placeholders if not already created
            if "progress_bar" not in st.session_state:
                st.session_state.progress_bar, st.session_state.status_text, st.session_state.log_container = display_training_progress(title=None)
            
            # Add a stop button
            if st.button("⛔ Stop Training", type="secondary"):
                st.session_state.training_status = "idle"
                st.rerun()
                
        elif st.session_state.training_status == "error":
            # Display error message
            st.error(f"### ❌ Training Error: {st.session_state.training_error}")
            
            # Provide specific advice based on common errors
            if "CUDA out of memory" in st.session_state.training_error or "GPU ran out of memory" in st.session_state.training_error:
                st.error("""
                💡 **Memory Issue Detected**: Your GPU ran out of memory. Try:
                1. Enabling 4-bit quantization (QLoRA) in the sidebar
                2. Reducing batch size to 1 or 2
                3. Using the smaller Gemma 2B model instead of 7B
                """)
            elif "No CUDA GPUs are available" in st.session_state.training_error:
                st.error("""
                💡 **GPU Issue Detected**: CUDA GPU not found. This app requires an NVIDIA GPU with CUDA support.
                Please check that your GPU drivers are installed correctly.
                """)
            elif "No such file or directory" in st.session_state.training_error:
                st.error("""
                💡 **File Error**: A required file could not be found. This may be due to a download error.
                Try restarting the application.
                """)
            elif "Mac compatibility issue" in st.session_state.training_error or "cadam32bit_grad_fp32" in st.session_state.training_error:
                st.error("""
                💡 **Mac Compatibility Issue**: Your Mac is not compatible with 8-bit quantization.
                
                Please go to Step 1 and ensure "Use 8-bit Quantization" is **disabled** in the sidebar.
                Then try again.
                """)
                
            # Reset button
            if st.button("🔄 Try Again", type="primary"):
                st.session_state.training_status = "idle"
                st.session_state.training_error = None
                st.rerun()
                
        elif st.session_state.training_status == "completed":
            # Show completion message
            st.success("### ✅ Training Complete!")
            st.balloons()
            st.success("🎉 Fine-tuning complete! Your personalized AI adapter is ready below.")
        
        else:
            # Start training button - more prominent
            # Disable button on non-CUDA systems unless in development/testing mode
            button_disabled = (DEVICE != "cuda") and not MAC_TESTING_MODE
            
            if button_disabled:
                st.button("🚀 Start Fine-Tuning", use_container_width=True, type="primary", disabled=True)
                st.error("Training requires an NVIDIA GPU. Your system does not have compatible hardware.")
            else:
                start_training = st.button("🚀 Start Fine-Tuning", use_container_width=True, type="primary")
                
                if start_training:
                    # Check if we're in Mac testing mode
                    if MAC_TESTING_MODE and DEVICE != "cuda":
                        st.warning("""
                        ### ⚠️ Mac Testing Mode Warning
                        
                        You're attempting to train on a Mac without NVIDIA GPU acceleration.
                        This will likely fail or be extremely slow.
                        
                        **This is unsupported and for experimentation only.**
                        """)
                        
                        # Removed the complex confirmation flow that was blocking execution
                        st.info("Proceeding with training in experimental Mac mode...")
                    
                    # Initialize progress elements
                    progress_bar, status_text, log_container = display_training_progress(title=None)
                    st.session_state.progress_bar = progress_bar
                    st.session_state.status_text = status_text
                    st.session_state.log_container = log_container
                    
                    # Set up output directory with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = os.path.join(config["output_dir"], f"gemma_lora_{timestamp}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Update config with new output directory
                    config["output_dir"] = output_dir
                    st.session_state.output_dir = output_dir
                    
                    # Update session state
                    st.session_state.training_status = "running"
                    
                    # Start training in a background thread
                    training_thread = threading.Thread(
                        target=train_in_background,
                        args=(
                            st.session_state.formatted_dataset,
                            config,
                            progress_bar,
                            status_text,
                            log_container
                        )
                    )
                    training_thread.daemon = True
                    training_thread.start()
                    
                    # Force a rerun to update UI
                    st.rerun()

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
        st.rerun()

if __name__ == "__main__":
    main()
