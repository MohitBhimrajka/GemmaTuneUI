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
            st.success(f"Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns.")
        elif file_format == 'jsonl':
            # Read JSONL file line by line
            data = []
            content = uploaded_file.getvalue().decode('utf-8')
            for line in content.splitlines():
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
            st.success(f"Successfully loaded JSONL file with {len(df)} rows and {len(df.columns)} columns.")
        elif file_format == 'json':
            # Read JSON array
            df = pd.read_json(uploaded_file)
            st.success(f"Successfully loaded JSON file with {len(df)} rows and {len(df.columns)} columns.")
        else:
            st.error(f"""
            ❌ Unsupported file format: {file_format}
            
            Please upload a CSV or JSONL file with your training examples.
            """)
            return None
            
        # Check if DataFrame is empty
        if df.empty:
            st.error("""
            ❌ The uploaded file appears to be empty.
            
            Please make sure your file contains data in the correct format.
            """)
            return None
            
        return df
    except Exception as e:
        error_message = str(e)
        
        # Provide more specific error messages for common issues
        if "Expecting ',' delimiter" in error_message:
            st.error("""
            ❌ CSV parsing error: The file doesn't seem to be a valid CSV file.
            
            Make sure the file uses commas to separate columns and each row is on a new line.
            """)
        elif "JSONDecodeError" in error_message or "Expecting value" in error_message:
            st.error("""
            ❌ JSON parsing error: The file doesn't seem to be valid JSON or JSONL.
            
            For JSONL files, each line should be a valid JSON object.
            For JSON files, the entire file should be a valid JSON array.
            """)
        elif "UnicodeDecodeError" in error_message:
            st.error("""
            ❌ Encoding error: Unable to read the file.
            
            Please make sure your file is saved with UTF-8 encoding.
            """)
        else:
            st.error(f"""
            ❌ Error loading file: {error_message}
            
            Please check your file format and try again.
            """)
        
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
    
    # Make a list of all possible columns to check for better user feedback
    all_acceptable_columns = []
    all_acceptable_columns.extend(column_variations["prompt_columns"])
    all_acceptable_columns.extend(column_variations["completion_columns"])
    all_acceptable_columns.extend(column_variations["text_columns"])
    
    # Display all columns for debugging
    st.caption(f"Your dataset contains these columns: {', '.join(df.columns)}")
    
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
        st.info(f"Using your existing '{text_col}' column as the formatted text.")
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
        st.info(f"Combining '{prompt_col}' and '{completion_col}' columns using Gemma's template.")
        
        # Show example of formatted text
        st.markdown("### Example of how your data will be formatted:")
        with st.expander("See formatted example", expanded=True):
            if len(df) > 0:
                example_input = df[prompt_col].iloc[0]
                example_output = df[completion_col].iloc[0]
                formatted_example = chat_template.format(
                    prompt=example_input,
                    completion=example_output
                )
                
                st.markdown("**Original Input:**")
                st.text(example_input)
                st.markdown("**Original Output:**")
                st.text(example_output)
                st.markdown("**Formatted for Gemma:**")
                st.code(formatted_example, language="text")
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
        
        prompt_examples = ", ".join([f"'{col}'" for col in column_variations["prompt_columns"][:3]])
        completion_examples = ", ".join([f"'{col}'" for col in column_variations["completion_columns"][:3]])
        text_examples = ", ".join([f"'{col}'" for col in column_variations["text_columns"][:3]])
        
        st.error(f"""
        ❌ **Column names not recognized in your dataset**
        
        Your dataset needs to have either:
        
        **Option 1: Two columns (recommended)**
        - One for user inputs/questions like {prompt_examples}
        - One for AI responses like {completion_examples}
        
        **Option 2: One pre-formatted column**
        - A single column like {text_examples} that's already formatted for conversation
        
        **Your current columns:** {', '.join(df.columns)}
        
        **Quick fix:** You can rename your columns in your file to match the expected names,
        or modify your CSV/JSONL to use these standard column names.
        """)
        return None
    
    # Format the DataFrame
    formatted_df = format_for_gemma(df, prompt_col, completion_col, text_col)
    
    if formatted_df is None:
        st.error("""
        ❌ Failed to format the dataset.
        
        Please check that your data has the correct columns and format.
        """)
        return None
    
    # Convert to datasets.Dataset
    dataset = Dataset.from_pandas(formatted_df)
    
    # Show formatting info
    if prompt_col and completion_col:
        st.success(f"""
        ✅ **Dataset formatted successfully!**
        
        - Found input column: '{prompt_col}'
        - Found response column: '{completion_col}'
        - Created formatted 'text' column for Gemma using the chat template
        """)
    elif text_col:
        st.success(f"""
        ✅ **Dataset formatted successfully!**
        
        - Using your pre-formatted '{text_col}' column
        - Renamed to 'text' for Gemma training
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
