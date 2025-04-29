AI323_RNN_Assignment
Simple RNN Implementation with PyTorch
Student Name: Ahmed Hany Mohamed Soliman HarfoushStudent ID: 4221382Course: AI323 - Computational NeuroscienceInstructor: Dr. Noha ElattarDate: April 29, 2025

Project Description
This repository contains the implementation of a simple Recurrent Neural Network (RNN) using PyTorch for a character-level language modeling task. The RNN predicts the next character in a sequence given a small text dataset. The implementation includes:

A custom SimpleRNN class with forward pass and backpropagation.
Data preparation using one-hot encoding for character sequences.
Training with the Adam optimizer and CrossEntropyLoss for 100 epochs.
Text generation functionality to produce 50 characters starting from a given seed text ("hello").
Proper handling of the computation graph to avoid errors during backpropagation (e.g., RuntimeError: Trying to backward through the graph a second time).

The code is implemented in simple_rnn.py and has been tested to run without errors.

Files

simple_rnn.py: The main Python script containing the RNN implementation, training loop, and text generation function.


Prerequisites
To run the code, you need the following dependencies:

Python 3.6 or higher
PyTorch (pip install torch)
NumPy (pip install numpy)


Installation

Clone this repository:
git clone https://github.com/yourusername/AI323_RNN_Assignment.git
cd AI323_RNN_Assignment


Install the required dependencies:
pip install torch numpy




Running the Code

Ensure the dependencies are installed.

Run the script:
python simple_rnn.py


Expected Output:

The script will train the RNN for 100 epochs, printing the average loss every 10 epochs.
After training, it will generate 50 characters of text starting with "hello".
Example output:Epoch [10/100], Loss: 2.3456
Epoch [20/100], Loss: 2.1234
...
Generated text: hellowor ld thi s is a simpl e rnn exa


Notes

The dataset is a repeated string ("hello world this is a simple rnn example "), so the generated text will reflect patterns in this input.
The RNN uses a hidden layer size of 128 and a sequence length of 5 characters.
The implementation ensures proper gradient management to avoid computation graph errors during backpropagation.
For improved text generation, you can experiment with increasing the hidden layer size, training for more epochs, or using a larger dataset (not required for the assignment).

