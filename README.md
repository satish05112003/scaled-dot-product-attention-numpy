# Scaled Dot-Product Attention from Scratch (NumPy)

## Project Overview
This project implements the Scaled Dot-Product Attention mechanism using pure NumPy and visualizes attention weights using Matplotlib.

The goal of this project is to understand how attention works internally without using deep learning frameworks like TensorFlow or PyTorch.

---

## What This Project Does
- Takes a sentence as input
- Creates random embeddings for words
- Generates Query (Q), Key (K), Value (V)
- Calculates Attention Scores
- Applies Scaling and Masking
- Applies Softmax to get Attention Weights
- Generates Output Attention Tensor
- Shows Attention Heatmap Visualization

---

## Full Attention Flow Diagram

![Full Flow Diagram](FULL_FLOW.png)

---

## Technologies Used
- Python
- NumPy
- Matplotlib

---

## Project Structure
```
attention.py
requirements.txt
README.md
FULL_FLOW.png
```

---

## How To Run

### Step 1 — Install Requirements
```
pip install -r requirements.txt
```

### Step 2 — Run Program
```
python attention.py
```

---

## Example Flow
Input Sentence → Tokenization → Embedding → QKV Creation →  
Score Calculation → Scaling → Mask → Softmax → Output → Heatmap

---

## Output You Will See
- Attention Output Tensor
- Word-to-word attention relation
- Attention Heatmap Graph

---

## Learning Outcome
After this project, you will understand:
- How attention works mathematically
- Why scaling is needed
- Why masking is used
- How transformers focus on important words
