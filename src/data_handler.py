"""
Functions for loading, validating, and processing data for Gemma fine-tuning.
"""

import pandas as pd
import streamlit as st
import os
import json
from datasets import Dataset
import io
from src.utils import get_column_name_variations, get_gemma_chat_template

def detect_file_format(uploaded_file):
    """
    Detect if the uploaded file is CSV or JSONL based on extension and content.
    
    Args:
        uploaded_file: The file uploaded through Streamlit.
        
    Returns:
        str: 'csv' or 'jsonl'
    """
    filename = uploaded_file.name.lower()
    
    if filename.endswith('.csv'):
        return 'csv'
    elif filename.endswith('.jsonl') or filename.endswith('.json'):
        # For JSON files, check if it's a JSONL file (one JSON object per line)
        content = uploaded_file.getvalue().decode('utf-8')
        uploaded_file.seek(0)  # Reset file pointer
        
        # If the file starts with '[' and ends with ']', it's likely a JSON array
        if content.strip().startswith('[') and content.strip().endswith(']'):
            return 'json'
        else:
            return 'jsonl'
    else:
        # Try to infer format from content
        content = uploaded_file.getvalue().decode('utf-8')
        uploaded_file.seek(0)  # Reset file pointer
        
        # Check if it looks like CSV
        if ',' in content.split('\n')[0]:
            return 'csv'
        # Check if each line is valid JSON
        try:
            for line in content.splitlines():
                if line.strip():  # Skip empty lines
                    json.loads(line)
            return 'jsonl'
        except json.JSONDecodeError:
            # Default to CSV if we can't determine
            return 'csv'

def load_dataframe(uploaded_file):
    """
    Load the uploaded file into a pandas DataFrame.
    
    Args:
        uploaded_file: The file uploaded through Streamlit.
        
    Returns:
        pd.DataFrame or None: The loaded DataFrame, or None if there was an error.
    """
    try:
        file_format = detect_file_format(uploaded_file)
        
        if file_format == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_format == 'jsonl':
            # Read JSONL file line by line
            data = []
            content = uploaded_file.getvalue().decode('utf-8')
            for line in content.splitlines():
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
        elif file_format == 'json':
            # Read JSON array
            df = pd.read_json(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_format}. Please upload a CSV or JSONL file.")
            return None
            
        # Check if DataFrame is empty
        if df.empty:
            st.error("The uploaded file appears to be empty.")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def identify_columns(df):
    """
    Identify the columns in the DataFrame that might be prompts, completions, or text.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        tuple: (prompt_col, completion_col, text_col)
    """
    column_variations = get_column_name_variations()
    
    # Initialize variables
    prompt_col = None
    completion_col = None
    text_col = None
    
    # Check for text column first
    for col in column_variations["text_columns"]:
        if col in df.columns:
            text_col = col
            break
    
    # Check for prompt column
    for col in column_variations["prompt_columns"]:
        if col in df.columns:
            prompt_col = col
            break
    
    # Check for completion column
    for col in column_variations["completion_columns"]:
        if col in df.columns:
            completion_col = col
            break
    
    # If no columns were found, try case-insensitive matching
    if not (prompt_col or completion_col or text_col):
        df_columns_lower = [col.lower() for col in df.columns]
        
        # Check for text column
        for col in column_variations["text_columns"]:
            if col.lower() in df_columns_lower:
                idx = df_columns_lower.index(col.lower())
                text_col = df.columns[idx]
                break
        
        # Check for prompt column
        for col in column_variations["prompt_columns"]:
            if col.lower() in df_columns_lower:
                idx = df_columns_lower.index(col.lower())
                prompt_col = df.columns[idx]
                break
        
        # Check for completion column
        for col in column_variations["completion_columns"]:
            if col.lower() in df_columns_lower:
                idx = df_columns_lower.index(col.lower())
                completion_col = df.columns[idx]
                break
    
    return prompt_col, completion_col, text_col

def format_for_gemma(df, prompt_col, completion_col, text_col):
    """
    Format the DataFrame for Gemma fine-tuning.
    
    Args:
        df: pandas DataFrame
        prompt_col: The column containing prompts
        completion_col: The column containing completions
        text_col: The column containing already formatted text
        
    Returns:
        pd.DataFrame: DataFrame with 'text' column formatted for Gemma
    """
    chat_template = get_gemma_chat_template()
    
    if text_col:
        # Dataset already has formatted text
        formatted_df = df.copy()
        formatted_df.rename(columns={text_col: 'text'}, inplace=True)
    elif prompt_col and completion_col:
        # Format using prompt and completion columns
        formatted_df = df.copy()
        formatted_df['text'] = formatted_df.apply(
            lambda row: chat_template.format(
                prompt=row[prompt_col], 
                completion=row[completion_col]
            ), 
            axis=1
        )
    else:
        return None
    
    # Keep only the necessary columns
    result_df = pd.DataFrame()
    result_df['text'] = formatted_df['text']
    
    return result_df

def load_and_format_dataset(uploaded_file):
    """
    Load and format the uploaded dataset for Gemma fine-tuning.
    
    Args:
        uploaded_file: The file uploaded through Streamlit.
        
    Returns:
        datasets.Dataset or None: The formatted dataset, or None if there was an error.
    """
    # Load the DataFrame
    df = load_dataframe(uploaded_file)
    if df is None:
        return None
    
    # Display the raw dataframe preview
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(5))
    
    # Identify columns
    prompt_col, completion_col, text_col = identify_columns(df)
    
    # Check if we found the necessary columns
    if not (prompt_col and completion_col) and not text_col:
        column_variations = get_column_name_variations()
        
        prompt_examples = ", ".join(column_variations["prompt_columns"][:3])
        completion_examples = ", ".join(column_variations["completion_columns"][:3])
        text_examples = ", ".join(column_variations["text_columns"][:3])
        
        st.error(f"""
        Could not identify the necessary columns in your dataset. 
        
        Your dataset should have either:
        - Both a prompt column (like {prompt_examples}) AND a completion column (like {completion_examples})
        - OR a single text column (like {text_examples}) that's already formatted for conversation
        
        Current columns found: {', '.join(df.columns)}
        """)
        return None
    
    # Format the DataFrame
    formatted_df = format_for_gemma(df, prompt_col, completion_col, text_col)
    
    if formatted_df is None:
        st.error("Failed to format the dataset.")
        return None
    
    # Convert to datasets.Dataset
    dataset = Dataset.from_pandas(formatted_df)
    
    # Show formatting info
    if prompt_col and completion_col:
        st.info(f"""
        **Detected Format:**
        - Prompt column: '{prompt_col}'
        - Completion column: '{completion_col}'
        
        Dataset has been formatted using Gemma's chat template:
        ```
        <start_of_turn>user
        [Your prompt]
        <end_of_turn>
        <start_of_turn>model
        [Your completion]
        <end_of_turn>
        ```
        """)
    elif text_col:
        st.info(f"""
        **Detected Format:**
        - Text column: '{text_col}'
        
        Using your existing formatted text column. Make sure it follows Gemma's format requirements.
        """)
    
    return dataset

def get_demo_dataset():
    """
    Creates a small, sample dataset suitable for Gemma fine-tuning.
    
    Returns:
        tuple: (dataset, raw_string) where dataset is a datasets.Dataset and 
               raw_string is the CSV string representation.
    """
    # Create sample data
    data = [
        {
            "prompt": "Explain what machine learning is to a 10-year-old.",
            "completion": "Machine learning is like teaching a computer to learn from examples, similar to how you learn. If I show you many pictures of cats and dogs, you'll learn the difference. Computers do the same thing! They look at lots of examples and learn patterns to make better guesses next time. That's how your video games get harder as you play, or how your phone suggests words when you type!"
        },
        {
            "prompt": "Write a short poem about the night sky.",
            "completion": "Silent diamonds in the night,\nWhispering stories of ancient light.\nMoon a guardian, gentle and bright,\nWatching over dreams until morning's sight."
        },
        {
            "prompt": "What are three tips for staying healthy?",
            "completion": "1. Drink plenty of water every day to stay hydrated.\n2. Get at least 30 minutes of physical activity daily.\n3. Eat a balanced diet with lots of fruits and vegetables."
        },
        {
            "prompt": "Summarize what photosynthesis is.",
            "completion": "Photosynthesis is how plants make their own food. They use sunlight, water, and carbon dioxide to create glucose (sugar) and oxygen. The sunlight provides energy, water comes from the roots, and carbon dioxide from the air. Chlorophyll, the green pigment in leaves, captures the sunlight energy for this process."
        },
        {
            "prompt": "How does recycling help the environment?",
            "completion": "Recycling helps the environment by reducing waste in landfills, conserving natural resources like timber and water, saving energy needed to produce new materials, reducing pollution from manufacturing processes, and decreasing greenhouse gas emissions. When we recycle paper, plastic, glass, and metals, we're giving these materials a new life instead of creating more waste."
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert to CSV string
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    
    # Format for Gemma
    formatted_df = format_for_gemma(df, "prompt", "completion", None)
    
    # Convert to datasets.Dataset
    dataset = Dataset.from_pandas(formatted_df)
    
    return dataset, csv_string
