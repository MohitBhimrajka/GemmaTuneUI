# GemmaTuneUI: Create Your Own Custom AI Assistant

✨ **An exceptionally user-friendly tool to personalize Google's Gemma AI models** ✨

GemmaTuneUI makes it easy for **anyone** - even if you're new to AI - to customize Gemma models to better follow your instructions, match your writing style, or focus on specific topics.

![GemmaTuneUI Main Interface](docs/images/main_interface.png)

## 🌟 What Can You Do With This?

- **Create a personalized AI assistant** that writes in your style
- **Make Gemma better at specific tasks** like summarizing articles, answering questions about a topic, or generating content
- **Teach the AI your preferences** by showing it examples of what you like
- **Experiment with AI customization** in a simple, guided interface

## 💡 Why GemmaTuneUI is Different

- **Super simple interface** with clear step-by-step guidance
- **Helpful explanations** of technical terms and settings
- **Runs completely on your computer** - your data stays private
- **Memory-efficient technology** (QLoRA) to work on consumer-grade GPUs
- **Downloadable adapter** that's tiny (just a few MB) compared to the full model

## 🖥️ What You'll Need

Before you start, make sure you have:

1. **Python 3.9 or newer** installed on your computer
2. **An NVIDIA GPU** with CUDA support:
   - **For Gemma 2B models:** At least 8GB GPU memory (RTX 3060 or better)
   - **For Gemma 7B models:** At least 14GB GPU memory (RTX 3080 or better)
3. **CUDA Toolkit** installed (version 11.8 or newer)
4. **About 5GB free disk space** for libraries and model files

Don't have a powerful enough GPU? Consider using a cloud service like Google Colab, vast.ai, or runpod.io with GPU access.

## 💻 System Requirements

- Python 3.9 or newer
- For training: NVIDIA GPU with at least 8GB VRAM (16GB+ recommended)
- For the web interface only: Any modern computer

### Mac Compatibility

GemmaTuneUI includes a special "Mac testing mode" that allows you to:
- Explore the full UI and workflow
- Test data loading and preprocessing
- Configure all parameters

However, actual model training requires an NVIDIA GPU with CUDA support, which is not available on Macs. When running on a Mac:

- The setup script will detect your Mac and enable testing mode automatically
- You'll receive clear warnings about limitations
- The app will attempt to use Metal Performance Shaders (MPS) on Apple Silicon Macs where possible
- You can still explore the interface and workflow, but model training will likely fail or be extremely slow

For full functionality, we recommend:
- Using a cloud service with NVIDIA GPUs (like Google Colab, vast.ai, or runpod.io)
- Setting up the application on a computer with an NVIDIA GPU

## ⚙️ Quick Installation

Just three simple steps:

1. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/GemmaTuneUI.git
   cd GemmaTuneUI
   ```

2. **Run the setup script:**
   
   On macOS/Linux:
   ```bash
   ./run.sh
   ```
   
   On Windows:
   ```
   run.bat
   ```
   
   This script will:
   - Create a virtual environment
   - Install all required dependencies
   - Launch the application
   
   *Note: The first run may take a few minutes to download and install dependencies.*

*Alternatively,* if you prefer manual setup:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:
   ```bash
   streamlit run app.py
   ```

The app will open in your web browser, ready to use!

## 📝 How to Use GemmaTuneUI

### Step 1: Configure Your Model
![Step 1: Model Configuration](docs/images/step1.png)

- Choose which Gemma model to customize (2B is faster, 7B may give better results)
- Adjust basic training settings or keep the defaults
- Everything has clear explanations - just hover over the (?) icons for help

### Step 2: Prepare Your Dataset
![Step 2: Upload Data](docs/images/step2.png)

You have two options:
- **Upload your own dataset** (CSV or JSONL file with examples)
- **Use our sample dataset** to try things out

### Step 3: Preview Your Data
![Step 3: Data Preview](docs/images/step3.png)

- See how your data will be formatted for training
- The app automatically detects the right columns
- Get confirmation that everything is ready

### Step 4: Train Your Custom AI
![Step 4: Training Process](docs/images/step4.png)

- Click the "Start Fine-Tuning" button
- Watch real-time progress as your AI learns
- This step takes time depending on your GPU and dataset (from minutes to hours)

### Step 5: Download Your Personalized AI
![Step 5: Download Result](docs/images/step5.png)

- See how well the AI learned with a simple loss chart
- Download your custom AI adapter (a small .zip file)
- Get instructions on how to use it

## 📊 Creating Your Own Dataset

Your dataset should include examples of what you want the AI to learn. It can be in CSV or JSONL format:

### CSV Format Example

```csv
prompt,completion
"Write a poem about the ocean","The endless blue stretches far,\nWaves dancing with timeless grace.\nSunlight sparkles on distant waters,\nAs seagulls soar in open space."
"Explain quantum computing to a child","Quantum computers are like magical calculators that can look at many answers at the same time! Regular computers can only look at one answer at a time, like checking boxes one by one. But quantum computers can check lots of boxes all at once, which makes them super fast for certain problems."
```

### JSONL Format Example

```jsonl
{"prompt": "Write a poem about the ocean", "completion": "The endless blue stretches far,\nWaves dancing with timeless grace.\nSunlight sparkles on distant waters,\nAs seagulls soar in open space."}
{"prompt": "Explain quantum computing to a child", "completion": "Quantum computers are like magical calculators that can look at many answers at the same time! Regular computers can only look at one answer at a time, like checking boxes one by one. But quantum computers can check lots of boxes all at once, which makes them super fast for certain problems."}
```

### Tips for Better Results

1. **Include 10-50 diverse examples** that represent what you want the AI to learn
2. **Be consistent** in your formatting and style
3. **Show don't tell** - demonstrate the style, tone, or knowledge you want
4. **Start simple** with a small dataset before making a large one

## 🔧 GPU Requirements Explained

Fine-tuning AI models requires a GPU with enough memory:

| Model | Minimum VRAM | Recommended | Example GPUs |
|-------|--------------|------------|--------------|
| Gemma 2B | 8GB | 10GB+ | RTX 3060, RTX 2070, RTX 3050 Ti |
| Gemma 7B | 14GB | 20GB+ | RTX 3090, RTX 4080, RTX A5000 |

The app uses QLoRA technology to reduce memory usage, but you still need a decent GPU.

## 🧩 Using Your Custom AI Adapter

After training, you'll download a small zip file (your adapter). Here's how to use it:

1. **Extract the zip file** to a folder on your computer
2. **Use this simple Python code:**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the original Gemma model (must match what you trained with)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",  # or whichever model you used
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

# Load your custom adapter
model = PeftModel.from_pretrained(model, "./adapter_directory")

# Try it out!
prompt = "Write a poem about mountains"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

The zip file includes a README with more detailed instructions and examples.

## ❓ Troubleshooting

**"CUDA out of memory" error:**
- Reduce batch size to 1
- Make sure 4-bit quantization is enabled
- Try the smaller Gemma 2B model
- Close other GPU-intensive applications

**"No CUDA GPUs available" error:**
- Make sure you have an NVIDIA GPU
- Install or update your NVIDIA drivers
- Install CUDA toolkit
- Verify with `nvidia-smi` command

**Other issues?**
- Check the [PEFT documentation](https://huggingface.co/docs/peft)
- Explore [Gemma examples](https://huggingface.co/google/gemma-7b)

## 🙏 Acknowledgments

- [Google's Gemma Model](https://github.com/google/gemma)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT Library](https://github.com/huggingface/peft)
- [Streamlit](https://streamlit.io/)

---

Made with ❤️ to make AI customization accessible to everyone
