# AI323_RNN_Assignment

## Simple RNN Implementation with PyTorch

**Student Name**: Ahmed Hany Mohamed Soliman Harfoush  
**Student ID**: 4221382  
**Course**: AI323 - Computational Neuroscience  
**Instructor**: Dr. Noha Elattar  
**Date**: April 29, 2025

---

### Project Description

This repository contains the implementation of a simple Recurrent Neural Network (RNN) using PyTorch for a character-level language modeling task. The RNN predicts the next character in a sequence given a small text dataset. The implementation includes:

- A custom `SimpleRNN` class with forward pass and backpropagation.
- Data preparation using one-hot encoding for character sequences.
- Training with the Adam optimizer and CrossEntropyLoss for 100 epochs.
- Text generation functionality to produce 50 characters starting from a given seed text ("hello").
- Proper handling of the computation graph to avoid errors during backpropagation (e.g., `RuntimeError: Trying to backward through the graph a second time`).

The code is implemented in `simple_rnn.py` and has been tested to run without errors.

---

### Files

- `simple_rnn.py`: The main Python script containing the RNN implementation, training loop, and text generation function.

---

### Prerequisites

To run the code, you need the following dependencies:

- Python 3.6 or higher
- PyTorch (`pip install torch`)
- NumPy (`pip install numpy`)

---

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/AI323_RNN_Assignment.git
   cd AI323_RNN_Assignment
   ```

2. Install the required dependencies:
   ```bash
   pip install torch numpy
   ```

---

### Running the Code

1. Ensure the dependencies are installed.
2. Run the script:
   ```bash
   python simple_rnn.py
   ```

3. **Expected Output**:
   - The script will train the RNN for 100 epochs, printing the average loss every 10 epochs.
   - After training, it will generate 50 characters of text starting with "hello".
   - Example output:
     ```
     Epoch [10/100], Loss: 2.3456
     Epoch [20/100], Loss: 2.1236
     ...
     Generated text: hellowor ld thi s is a simpl e rnn exam
     ```

---

### Notes

- The dataset is a repeated string ("hello world this is a simple rnn example "), so the generated text will reflect patterns in this input.
- The RNN uses a hidden layer size of 128 and a sequence length of 5 characters.
- The implementation ensures proper gradient management to avoid computation graph errors during backpropagation.
- For improved text generation, you can experiment with increasing the hidden layer size, training for more epochs, or using a larger dataset (not required for the assignment).

---

### Contact

For any questions or clarifications, contact the instructor, Dr. Noha Elattar, or the student, Ahmed Hany Mohamed Soliman Harfoush.
