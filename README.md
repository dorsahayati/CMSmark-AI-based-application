# CMSmark AI-based classifier Application

## Overview
CMSmark is a graphical user interface (GUI) application designed for performing classification tasks using machine learning models. The application allows users to input CSV data, set a probability threshold, and select a model type (excellent, good, or excellent-good) for inference. It generates various results and visualizations based on the model predictions.

## Features
- User-friendly interface for uploading CSV files.
- Ability to set a threshold for classification.
- Selection of model types for inference.
- Generation of classification results and visualizations, including plots and heatmaps.

## Project Structure
```
python-gui-app/
├── src/
│   ├── main.py              # Entry point for the GUI application
│   ├── cmsmark.py           # Core logic for model inference and plotting
│   ├── gui/
│   │   └── app_window.py     # GUI components and event handling
│   └── utils/
│       └── file_utils.py     # Utility functions for file handling
├── requirements.txt          # Project dependencies
├── build.ps1                 # PowerShell script for building the application
└── CMSmark.spec              # PyInstaller specification file
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd python-gui-app
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```
   python src/main.py
   ```

2. Use the GUI to:
   - Upload your CSV data file.
   - Set the desired probability threshold.
   - Select the model type for inference.
   - Click on the "Run Inference" button to generate results and visualizations.
  
<img width="817" height="537" alt="image" src="https://github.com/user-attachments/assets/58032c86-b799-489d-b738-484bf7d37c38" />





## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
