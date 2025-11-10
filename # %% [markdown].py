# %% [markdown]
# # Initial Analysis of the CMU Pronouncing Dictionary (CMUdict)
# 
# In this notebook, we begin our project by performing an initial analysis of the CMUdict dataset. We will:
# - Load the CMUdict dictionary
# - Analyze the distribution of phones
# - Summarize key statistics (number of words, unique phones, phone frequencies)
# 
# This will help us understand the complexity of the problem and inform the design of our baseline model.

# %%
# Load and analyze the CMUdict dictionary
import collections
import matplotlib.pyplot as plt

# Load the dictionary file
cmudict_path = 'cmudict.dict.txt'
words = []
phones = []
phone_freq = collections.Counter()

with open(cmudict_path, 'r') as f:
    for line in f:
        if line.strip() and not line.startswith(';;;'):
            parts = line.strip().split()
            word = parts[0]
            word_phones = parts[1:]
            words.append(word)
            phones.extend(word_phones)
            phone_freq.update(word_phones)

num_words = len(words)
unique_phones = set(phones)
num_unique_phones = len(unique_phones)

print(f"Number of words: {num_words}")
print(f"Number of unique phones: {num_unique_phones}")
print(f"Most common phones: {phone_freq.most_common(10)}")

# Plot phone frequency distribution
plt.figure(figsize=(12,4))
plt.bar(*zip(*phone_freq.most_common(20)))
plt.title('Top 20 Most Common Phones in CMUdict')
plt.ylabel('Frequency')
plt.xlabel('Phone')
plt.show()

# %%
# Calculate total phonemes and percentages
total_phonemes = sum(phone_freq.values())
print(f"\nTotal number of phonemes: {total_phonemes}")

# Calculate and plot phone frequencies as percentages
percentages = {phone: (count/total_phonemes)*100 for phone, count in phone_freq.items()}
sorted_percentages = dict(sorted(percentages.items(), key=lambda x: x[1], reverse=True))

plt.figure(figsize=(12,4))
plt.bar(list(sorted_percentages.keys())[:20], list(sorted_percentages.values())[:20])
plt.title('Top 20 Most Common Phones in CMUdict (% of Total)')
plt.ylabel('Percentage of Total Phones')
plt.xlabel('Phone')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print top 10 phones with percentages
print("\nTop 10 phones by percentage:")
for phone, pct in list(sorted_percentages.items())[:10]:
    print(f"{phone}: {pct:.2f}%")

# %% [markdown]
# ## Observations from Initial Analysis
# 
# - The CMUdict contains a large number of words and a diverse set of phones.
# - Some phones are much more common than others, indicating an imbalanced distribution.
# - This imbalance may affect model performance and bias.
# 
# Next, we will develop a simple baseline classifier using phone statistics.

# %% [markdown]
# ## Baseline Classifier: Predicting Word Length from Phone Count
# 
# As a simple baseline, we will build a regressor that predicts the number of letters in a word (word length) from the number of phones in its pronunciation. This will help us understand the relationship between orthography and phonology in the dataset.

# %%
# Baseline regressor: predict word length from phone count
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Prepare data: word length vs. phone count
X = []  # phone counts
y = []  # word lengths
for word, line in zip(words, open(cmudict_path)):
    if line.strip() and not line.startswith(';;;'):
        parts = line.strip().split()
        word_phones = parts[1:]
        X.append([len(word_phones)])
        y.append(len(word))

X = np.array(X)
y = np.array(y)

# Use a subset for demonstration
X_sub = X[:1000]
y_sub = y[:1000]

# Fit linear regression
reg = LinearRegression().fit(X_sub, y_sub)
y_pred = reg.predict(X_sub)

print(f"Baseline regressor coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")
print(f"MSE: {mean_squared_error(y_sub, y_pred):.2f}")
print(f"R^2: {r2_score(y_sub, y_pred):.2f}")

# Plot predictions vs. true values
plt.figure(figsize=(6,4))
plt.scatter(X_sub, y_sub, alpha=0.3, label='True')
plt.plot(X_sub, y_pred, color='red', label='Prediction')
plt.xlabel('Phone Count')
plt.ylabel('Word Length')
plt.title('Baseline: Word Length vs. Phone Count')
plt.legend()
plt.show()

# %% [markdown]
# ## Baseline Results and Discussion
# 
# - The baseline regressor provides a simple mapping from phone count to word length.
# - The $R^2$ and MSE values give us a sense of how much variance in word length can be explained by phone count alone.
# - This baseline helps us understand the inherent bias and limitations of using only phone statistics for prediction.
# 
# **Next steps:**
# - Explore more complex features (e.g., phone types, stress patterns)
# - Develop more advanced models
# - Investigate sources of bias in the dataset and model predictions

# %% [markdown]
# ---
# 
# ### Proof of Project Initiation
# 
# - The code and analysis above demonstrate that we have begun the project in a significant manner.
# - We have performed initial data exploration, implemented a baseline model, and documented our findings.
# - All code is available in this notebook for reproducibility and further development.

# %%
# Find words with 3 letters and 5 phones
three_letter_five_phones = []
for word, line in zip(words, open(cmudict_path)):
    if len(word) == 3:  # 3 letters
        parts = line.strip().split()
        word_phones = parts[1:]
        if len(word_phones) == 5:  # 5 phones
            three_letter_five_phones.append((word, word_phones))

print("\nExamples of 3-letter words with 5 phones:")
for word, phones in three_letter_five_phones[:20]:  # Show first 3 examples
    print(f"Word: {word}, Phones: {' '.join(phones)}")

# %%
def padding_word(seq, max_length):
    padded_seq = seq + '%' + '-' * (max_length - len(seq)-1)
    return padded_seq[:max_length]

print(padding_word('cat', 4))

def phone_padding(seq, max_length):
    padded_seq = list(seq)
    for i in range(len(seq), max_length):
        padded_seq.append( '-')
    return padded_seq[:max_length]
print(phone_padding(['K', 'AE1', 'T'], 6))

# %%
# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Create character and phoneme vocabularies
def create_vocab(words, phones):
    char_vocab = set(''.join(words))
    phone_vocab = set(phone for phone_list in phones for phone in phone_list)
    
    # Create mapping dictionaries
    char_to_idx = {char: idx for idx, char in enumerate(sorted(char_vocab))}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    phone_to_idx = {phone: idx for idx, phone in enumerate(sorted(phone_vocab))}
    idx_to_phone = {idx: phone for phone, idx in phone_to_idx.items()}
    
    return char_to_idx, idx_to_char, phone_to_idx, idx_to_phone



# Prepare dataset
word_phone_pairs = []
for word, line in zip(words, open(cmudict_path)):
    if line.strip() and not line.startswith(';;;'):
        parts = line.strip().split()
        word_phones = parts[1:]

        # Add padding Atul 
        max_length = len(word) + 1
        #+ len(word_phones)
        word_phone_pairs.append((padding_word(word, max_length), phone_padding(word_phones, max_length)))
        #word_phone_pairs.append((word, phone_padding(word_phones, max_length)))
        # word_phone_pairs.append((word, word_phones))

# Create vocabularies
char_to_idx, idx_to_char, phone_to_idx, idx_to_phone = create_vocab(
    [pair[0] for pair in word_phone_pairs],
    [pair[1] for pair in word_phone_pairs]
)

print(f"Vocabulary sizes: {len(char_to_idx)} characters, {len(phone_to_idx)} phones")

# %%
# Custom Dataset class for sliding window approach
class PhonemeDataset(Dataset):
    def __init__(self, word_phone_pairs, char_to_idx, phone_to_idx, window_size=7):
        self.window_size = window_size
        self.data = []
        
        # Create sliding windows for each word
        for word, phones in word_phone_pairs:
            # Pad word with spaces
            padded_word = ' ' * (window_size//2) + word + ' ' * (window_size//2)

            
            
            # Create windows and their corresponding phonemes
            for i in range(len(word)):
          
                
                window = padded_word[i:i+window_size]
                # Convert characters to indices
                char_indices = [char_to_idx.get(c, 0) for c in window]
                # Convert phoneme to index
                if i < len(phones):
                    phone_idx = phone_to_idx.get(phones[i], 0)
                    self.data.append((char_indices, phone_idx))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        chars, phone = self.data[idx]
        return torch.tensor(chars), torch.tensor(phone)

# Split data into train and test sets
train_pairs, test_pairs = train_test_split(word_phone_pairs, test_size=0.2, random_state=42)

# Create datasets
train_dataset = PhonemeDataset(train_pairs, char_to_idx, phone_to_idx)
test_dataset = PhonemeDataset(test_pairs, char_to_idx, phone_to_idx)

print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# %%
# Define the neural network model
class PhonemeClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_phones):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size * 7, hidden_size)  # 7 is window size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_phones)
    
    def forward(self, x):
        # x shape: (batch_size, window_size)
        x = self.embedding(x)  # (batch_size, window_size, hidden_size)
        x = x.view(x.size(0), -1)  # Flatten the window dimension
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model, loss function, and optimizer
hidden_size = 128
model = PhonemeClassifier(len(char_to_idx), hidden_size, len(phone_to_idx))

# pad_index = phone_to_idx['-']
# criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

print("Model initialized")
print(phone_to_idx)

# %%
# Training loop
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_chars, batch_phones in train_loader:
        # Move batch to device
        batch_chars = batch_chars.to(device)
        batch_phones = batch_phones.to(device)
        
        # Forward pass
        outputs = model(batch_chars)
        loss = criterion(outputs, batch_phones)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_chars, batch_phones in test_loader:
            batch_chars = batch_chars.to(device)
            batch_phones = batch_phones.to(device)
            outputs = model(batch_chars)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_phones.size(0)
            correct += (predicted == batch_phones).sum().item()
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Average Loss: {total_loss/len(train_loader):.4f}')
    print(f'Test Accuracy: {100 * correct/total:.2f}%\n')

# %%
# Test the model on a few examples
def predict_phonemes(word, model, char_to_idx, idx_to_phone, window_size=7):
    model.eval()
    # Pad the word
    padded_word = ' ' * (window_size//2) + word + ' ' * (window_size//2)
    predictions = []
    
    with torch.no_grad():
        for i in range(len(word)):
            window = padded_word[i:i+window_size]
            char_indices = torch.tensor([char_to_idx.get(c, 0) for c in window]).unsqueeze(0)
            char_indices = char_indices.to(device)
            
            output = model(char_indices)
            _, predicted = torch.max(output.data, 1)
            phoneme = idx_to_phone[predicted.item()]
            if phoneme == "%":
                break
            predictions.append(idx_to_phone[predicted.item()])
    
    # return predictions
    return remove_consecutive_duplicates(predictions)

def remove_consecutive_duplicates(phonemes):
    result = []
    for p in phonemes:
        if not result or p != result[-1]:
            result.append(p)
    return result

def remove_padding(phonemes, pad_token='-'):
    return [p for p in phonemes if p != pad_token]
# Test a few words
test_words = ['cat', 'dog', 'bird', 'apple', 'aple', 'bell', 'banana']
print("Example predictions:")
for word in test_words:
    if word in [pair[0] for pair in word_phone_pairs]:
        # predicted = predict_phonemes(word, model, char_to_idx, idx_to_phone)

        predicted = remove_padding(predict_phonemes(word, model, char_to_idx, idx_to_phone))
        print(f"{word}: {' '.join(predicted)}")

# %%
# Test the model and analyze predictions
def analyze_prediction(word, model, char_to_idx, idx_to_phone, word_phone_pairs):
    predicted = predict_phonemes(word, model, char_to_idx, idx_to_phone)
    print(f"\nWord: {word}")
    print(f"Predicted: {' '.join(predicted)}")
    
    # Find actual pronunciation if word exists in dictionary
    actual = None
    for w, phones in word_phone_pairs:
        if w == word:
            actual = phones
            break
    
    if actual:
        print(f"Actual: {' '.join(actual)}")
        print(f"Match: {'Yes' if predicted == actual else 'No'}")
    else:
        print("Actual: Not in dictionary (unknown word)")

# Test both known and unknown words
# test_words = ['cat', 'apple', 'aple', 'Zeitgeist', 'zeitgeist', 'quizzaciously']
test_words = ['cat', 'apple', 'aple', 'Zeitgeist', 'zeitgeist', 'quizzaciously']
for word in test_words:
    analyze_prediction(word, model, char_to_idx, idx_to_phone, word_phone_pairs)

# %% [markdown]
# ### Understanding the Model's Behavior
# 
# 1. **Unknown Words**: The model can now make predictions for words not in the dictionary, like 'aple'
# 2. **Duplicate Phonemes**: The duplicate phonemes (like 'D D' in 'bird') occur because:
#    - The model predicts each phoneme independently
#    - It doesn't have context about previously predicted phonemes
#    - There's no constraint to prevent duplicate predictions
# 
# 3. **Potential Improvements**:
#    - Add sequence modeling (LSTM/RNN) to consider previous predictions
#    - Include phonotactic constraints
#    - Use beam search for better sequence prediction
#    - Add attention mechanism for better context handling
# 
# The model currently treats each position independently, which can lead to these duplications. A more sophisticated sequence-to-sequence model would likely perform better.

# %%
# After training finishes (on your local machine)
torch.save(model.state_dict(), "nettalk_state_dict.pt")
print("Saved state_dict to nettalk_state_dict.pt")


# %%
import json
with open("phone_vocab.json", "w") as f:
    json.dump(phone_to_idx, f)


