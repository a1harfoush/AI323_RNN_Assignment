import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the RNN class
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Input to hidden layer transformation
        self.Wxh = nn.Linear(input_size, hidden_size)
        # Hidden to hidden transformation
        self.Whh = nn.Linear(hidden_size, hidden_size)
        # Hidden to output transformation
        self.Why = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, hidden):
        # x shape: (batch_size, input_size)
        # hidden shape: (batch_size, hidden_size)
        
        # Compute hidden state
        hidden = self.tanh(self.Wxh(x) + self.Whh(hidden))
        
        # Compute output
        output = self.Why(hidden)
        output = self.softmax(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

# Data preparation
def prepare_data(text, seq_length):
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Create input-output pairs
    inputs = []
    targets = []
    for i in range(0, len(text) - seq_length):
        inputs.append([char_to_idx[ch] for ch in text[i:i+seq_length]])
        targets.append(char_to_idx[text[i+seq_length]])
    
    return inputs, targets, char_to_idx, idx_to_char, len(chars)

# Training function (corrected)
def train(model, inputs, targets, vocab_size, num_epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        
        for i, (input_seq, target) in enumerate(zip(inputs, targets)):
            # Convert to one-hot encoding
            input_tensor = torch.zeros((len(input_seq), vocab_size))
            for t, char_idx in enumerate(input_seq):
                input_tensor[t][char_idx] = 1
            input_tensor = input_tensor.to(device)
            target_tensor = torch.tensor([target], dtype=torch.long).to(device)
            
            # Initialize hidden state
            hidden = model.init_hidden(1).to(device)
            optimizer.zero_grad()
            
            # Forward pass through sequence
            for t in range(len(input_seq)):
                output, hidden = model(input_tensor[t].unsqueeze(0), hidden)
                # Compute loss only for the last output
                if t == len(input_seq) - 1:
                    loss = criterion(output, target_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(inputs):.4f}')

# Main execution
if __name__ == "__main__":
    # Sample text data
    text = "hello world this is a simple rnn example " * 10
    seq_length = 5
    
    # Prepare data
    inputs, targets, char_to_idx, idx_to_char, vocab_size = prepare_data(text, seq_length)
    
    # Model parameters
    input_size = vocab_size
    hidden_size = 128
    output_size = vocab_size
    
    # Initialize model
    model = SimpleRNN(input_size, hidden_size, output_size)
    
    # Train model
    train(model, inputs, targets, vocab_size)
    
    # Generate sample text
    def generate_text(model, start_text, length, char_to_idx, idx_to_char, vocab_size):
        model.eval()
        hidden = model.init_hidden(1)
        input_text = start_text[-seq_length:]
        generated = start_text
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for _ in range(length):
            # Prepare input
            input_seq = [char_to_idx[ch] for ch in input_text]
            input_tensor = torch.zeros((len(input_seq), vocab_size))
            for t, char_idx in enumerate(input_seq):
                input_tensor[t][char_idx] = 1
            input_tensor = input_tensor.to(device)
            hidden = hidden.to(device)
            
            # Forward pass
            with torch.no_grad():
                output, hidden = model(input_tensor[-1].unsqueeze(0), hidden)
            
            # Sample next character
            probs = output.squeeze().cpu().numpy()
            char_idx = np.random.choice(range(vocab_size), p=probs)
            next_char = idx_to_char[char_idx]
            
            generated += next_char
            input_text = input_text[1:] + next_char
        
        return generated
    
    # Generate sample output
    start_text = "hello"
    generated_text = generate_text(model, start_text, 50, char_to_idx, idx_to_char, vocab_size)
    print(f"\nGenerated text: {generated_text}")