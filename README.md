# Ripley's K Function Analysis Tool

This repository contains a demonstration of Ripley's K function, a spatial statistics tool used to analyze point patterns and determine whether they exhibit clustering, randomness, or regular dispersion.

## Contents

- `ripley_k_demonstration.md`: Detailed explanation of Ripley's K function with mathematical definitions, implementation, and interpretation
- `ripley_k_analysis.py`: Python implementation of the Ripley's K function analysis
- `requirements.txt`: List of required Python packages

## Setup

1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Analysis

Execute the Python script to generate the visualizations:

```bash
python ripley_k_analysis.py
```

This will generate three image files:
- `point_patterns.png`: Visualization of the three different point patterns
- `ripley_k_plot.png`: Plots of the K and L functions for each pattern
- `ripley_k_interpretation.png`: Visual guide for interpreting the L function results

## Understanding the Results

Please refer to the detailed explanation in `ripley_k_demonstration.md` for:
- Mathematical definition of Ripley's K function
- Explanation of the data generation process
- Interpretation of the K and L function plots
- Applications and further extensions

## Requirements

- Python 3.7 or higher
- NumPy
- Matplotlib
- SciPy
- Seaborn 