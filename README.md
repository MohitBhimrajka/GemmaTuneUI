# GemmaTuneUI: User-Friendly AI Fine-Tuning

✨ An exceptionally user-friendly web application for fine-tuning Google's Gemma models! ✨

GemmaTuneUI makes it easy for users **without deep machine learning knowledge** to customize Google's powerful Gemma large language models to better match their specific needs, writing style, or task requirements.

![GemmaTuneUI Main Interface](docs/images/main_interface.png)

## 🌟 Features

- **User-Friendly Interface**: Step-by-step workflow with clear explanations
- **QLoRA Fine-Tuning**: Memory-efficient Parameter-Efficient Fine-Tuning (PEFT) method
- **Support for All Gemma Models**: Compatible with 2B and 7B variants
- **Simple Data Format**: Easy CSV or JSONL input with automatic column detection
- **Sample Dataset**: Included example data for testing
- **System Compatibility Check**: Automatic GPU detection and compatibility testing
- **Visual Progress Tracking**: Real-time training progress with loss visualization
- **Downloadable Adapters**: Export small, portable LoRA adapters to use with the base model

## 🚀 Why Use GemmaTuneUI?

- **Create a Personalized AI Assistant**: Fine-tune Gemma to match your writing style or specific domain knowledge
- **Build Task-Specific AI**: Customize Gemma for summarization, content generation, Q&A, or creative writing
- **Experiment with AI Customization**: Learn about model fine-tuning with a user-friendly interface
- **Run Locally**: All processing happens on your local machine - no data sent to external services

## 🖥️ Prerequisites

Before running GemmaTuneUI, make sure you have:

1. **Python 3.9+** installed on your system
2. **NVIDIA GPU** with CUDA support:
   - RTX 3060 or better recommended for Gemma 2B models
   - RTX 3080/4070 or better recommended for Gemma 7B models
   - At least 8GB VRAM for 2B models with quantization
   - At least 14GB VRAM for 7B models with quantization
3. **CUDA Toolkit** (11.8+ recommended) and matching PyTorch installation
4. **~5GB free disk space** for libraries and model files

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/GemmaTuneUI.git
   cd GemmaTuneUI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📝 Using GemmaTuneUI

### Step 1: Configure Your Model & Training
![Step 1: Configuration](docs/images/step1.png)

Use the sidebar to:
- Select the Gemma model version
- Enable/disable quantization
- Adjust training parameters (epochs, learning rate, batch size)
- Configure advanced LoRA settings

### Step 2: Upload Your Data
![Step 2: Upload Data](docs/images/step2.png)

Either:
- Upload your CSV or JSONL dataset
- Use the provided sample dataset for testing

### Step 3: Preview & Format Data
![Step 3: Preview Data](docs/images/step3.png)

The application will:
- Show a preview of your data
- Automatically detect prompt and completion columns
- Format your data for Gemma fine-tuning
- Display the number of examples loaded

### Step 4: Start Fine-Tuning
![Step 4: Fine-Tuning](docs/images/step4.png)

Click the "Start Fine-Tuning" button to:
- Load the selected Gemma model with quantization if enabled
- Apply LoRA configuration for efficient fine-tuning
- Begin training with real-time progress updates

### Step 5: Get Your Custom AI Adapter
![Step 5: Download Results](docs/images/step5.png)

After training completes:
- View a loss plot showing training progress
- Download your LoRA adapter as a zip file
- Use the included README for instructions on using your adapter

## 📊 Dataset Format

Your dataset should be in one of these formats:

### CSV Format

```csv
prompt,completion
"Explain what machine learning is to a 10-year-old.","Machine learning is like teaching a computer to learn from examples, similar to how you learn. If I show you many pictures of cats and dogs, you'll learn the difference. Computers do the same thing!"
"Write a short poem about the night sky.","Silent diamonds in the night,\nWhispering stories of ancient light.\nMoon a guardian, gentle and bright,\nWatching over dreams until morning's sight."
```

### JSONL Format

```jsonl
{"prompt": "Explain what machine learning is to a 10-year-old.", "completion": "Machine learning is like teaching a computer to learn from examples, similar to how you learn. If I show you many pictures of cats and dogs, you'll learn the difference. Computers do the same thing!"}
{"prompt": "Write a short poem about the night sky.", "completion": "Silent diamonds in the night,\nWhispering stories of ancient light.\nMoon a guardian, gentle and bright,\nWatching over dreams until morning's sight."}
```

### Column Name Variations

The application automatically detects these common column names:

- **Prompt columns**: prompt, instruction, input, question, context, user_input
- **Completion columns**: completion, response, output, answer, assistant, model_output
- **Text columns** (if already formatted): text, content, message

## 🔧 Hardware Requirements

Fine-tuning large language models requires GPU acceleration:

| Model | Minimum VRAM (with 4-bit) | Recommended VRAM | Example GPU |
|-------|----------------------------|------------------|-------------|
| Gemma 2B | 8GB | 12GB+ | RTX 3060, RTX 2080 |
| Gemma 7B | 14GB | 24GB+ | RTX 3090, RTX 4090 |

## 🧩 Using Your Fine-Tuned Adapter

After downloading your adapter zip file:

1. Extract the files to a directory
2. Load both the original Gemma model and your adapter:

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model and tokenizer
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")  # or whichever model you used
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

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

## 🛠️ Tech Stack

- **Streamlit**: For the user interface
- **Transformers**: For model loading and tokenization
- **PEFT**: For Parameter-Efficient Fine-Tuning with LoRA/QLoRA
- **bitsandbytes**: For 4-bit quantization
- **torch**: PyTorch backend
- **datasets**: For data handling
- **accelerate**: For efficient hardware utilization
- **plotly**: For visualizing training progress

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Google's Gemma Model](https://github.com/google/gemma)
- [Hugging Face Transformers Library](https://github.com/huggingface/transformers)
- [PEFT Library](https://github.com/huggingface/peft)
- [Streamlit](https://streamlit.io/)

---

Made with ❤️ to make AI fine-tuning accessible to everyone
