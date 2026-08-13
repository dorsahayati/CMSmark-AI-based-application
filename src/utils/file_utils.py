def load_csv_file(file_path):
    """Load a CSV file and return its contents as a pandas DataFrame."""
    import pandas as pd
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def save_results_to_csv(results_df, output_path):
    """Save the results DataFrame to a CSV file."""
    try:
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

def validate_csv_file(file_path):
    """Check if the CSV file exists and is valid."""
    import os
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return False
    return True

def get_file_extension(file_path):
    """Return the file extension of the given file path."""
    import os
    return os.path.splitext(file_path)[1]