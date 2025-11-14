# LSTM + CTC Grapheme-to-Phoneme (G2P) System

A deep learning system for converting written text (graphemes) into phonetic representations (phonemes) using a Bidirectional LSTM with Connectionist Temporal Classification (CTC) loss.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Why This Project Exists](#why-this-project-exists)
3. [Dataset: CMU Pronouncing Dictionary](#dataset-cmu-pronouncing-dictionary)
4. [Architecture & Components](#architecture--components)
5. [File Structure](#file-structure)
6. [Detailed Component Breakdown](#detailed-component-breakdown)
7. [Installation & Setup](#installation--setup)
8. [Usage](#usage)
9. [Results & Performance](#results--performance)
10. [Technical Deep Dive](#technical-deep-dive)
11. [Working Demo With Syntheziser Model](#Synthesizer-Demo) 

---

## Project Overview

This system learns to pronounce English words by training on the CMU Pronouncing Dictionary (CMUdict), which contains ~135,000 word-pronunciation pairs. Unlike simple character-by-character mapping, this model uses a sequence-to-sequence approach that can handle:

- **Variable-length mappings**: Words and their phoneme sequences aren't always the same length
- **Complex phonetic patterns**: Silent letters, combined sounds, irregular pronunciations
- **Context-aware predictions**: Understanding how surrounding letters affect pronunciation

**Example predictions:**
- `google` → `G UW G AH L`
- `physics` → `F IH Z IH K S`
- `algorithm` → `AE L G ER IH DH AH M`

---

## Why This Project Exists

### Problem Statement
Converting written text to phonetic representations is fundamental for:
- **Text-to-Speech (TTS) systems**: Computers need to know how to pronounce words
- **Speech Recognition**: Understanding phonetic patterns helps recognize spoken words
- **Language Learning Tools**: Helping non-native speakers learn pronunciation
- **Accessibility**: Enabling screen readers and assistive technologies

### Why Not Simple Rules?
English pronunciation is notoriously irregular. Simple rule-based systems fail because:
- Same letters sound different: "ough" in rough, though, through, thought
- Silent letters: knight, pneumonia, debt
- Context matters: "read" (present) vs "read" (past)
- Foreign words: chaos, yacht, café

### The Deep Learning Advantage
A neural network learns these patterns automatically from data, handling exceptions and irregularities without explicit programming.

---

## Dataset: CMU Pronouncing Dictionary

**Why CMUdict?**
- **Comprehensive**: ~135,000 entries covering most English words
- **Standardized**: Uses ARPAbet phoneme notation (39 phonemes)
- **Stress markers**: Indicates syllable emphasis (removed in this implementation for simplicity)
- **Multiple pronunciations**: Some words have variant entries (e.g., "read(1)", "read(2)")

**Format Example:**
```
GOOGLE  G UW1 G AH0 L
PHYSICS  F IH1 Z IH0 K S
ALGORITHM  AE1 L G ER0 IH0 DH AH0 M
```

**Preprocessing:**
1. Strip stress markers: `UW1` → `UW`
2. Remove variant indicators: `READ(2)` → `READ`
3. Lowercase words for consistency
4. Build vocabulary mappings for characters and phonemes

---

## Architecture & Components

### Model Architecture: Bidirectional LSTM + CTC

```
Input Word (characters)
        ↓
    Embedding Layer (64-dim)
        ↓
Bidirectional LSTM (256 hidden units × 2 directions)
        ↓
    Linear Layer (to phoneme vocabulary size)
        ↓
    Log Softmax
        ↓
   CTC Loss / Decoding
        ↓
Output Phonemes
```

**Why Bidirectional?**
- Reads the word both forward and backward
- Captures context from both directions: "knight" needs to know 'k' is followed by 'n' (silent) and preceded by nothing

**Why CTC (Connectionist Temporal Classification)?**
- Solves the alignment problem: we don't know which characters map to which phonemes
- Allows variable-length inputs and outputs
- Automatically learns to insert/skip phonemes as needed
- Uses a special "blank" token to handle repetitions and gaps

---

## File Structure

```
├── LSTMtesting.ipynb           # Main training & evaluation notebook
├── cmudict.dict.txt            # CMU Pronouncing Dictionary data
├── cmudict.phones              # Phoneme type reference (vowel/consonant/etc.)
├── lstm_ctc_g2p.pth            # Trained model checkpoint
├── char_vocab_LSTM.json        # Character → index mapping
├── phone_vocab_LSTM.json       # Phoneme → index mapping
├── char_vocab.json             # Alternative vocab (with stress)
├── phone_vocab.json            # Alternative vocab (with stress)
└── README.md                   # This file
```

---

## Detailed Component Breakdown

### 1. **Data Loading (`load_cmudict` function)**

**Purpose:** Parse the CMUdict text file into usable Python data structures.

**What it does:**
```python
def load_cmudict(path, max_words=None):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith(";;;"):
                continue  # Skip empty lines and comments
            parts = line.strip().split()
            word = re.sub(r"\(\d+\)$", "", parts[0]).lower()  # Remove (1), (2) variants
            phones = parts[1:]
            if STRIP_STRESS:
                phones = [re.sub(r"\d+$", "", p) for p in phones]  # Remove stress digits
            entries.append((word, phones))
```

**Why it's needed:** Raw dictionary files aren't ready for neural networks. We need clean (word, phoneme_list) tuples.

---

### 2. **Vocabulary Building**

**Character Vocabulary:**
```python
graphemes = sorted({c for w, _ in entries for c in w})
graphemes = ["<pad>", "<s>", "</s>"] + graphemes
char2idx = {c: i for i, c in enumerate(graphemes)}
```

**Why special tokens?**
- `<pad>`: Padding for batches (makes sequences same length)
- `<s>`, `</s>`: Start/end markers (not used in this implementation but common practice)

**Phoneme Vocabulary:**
```python
phones = sorted({p for _, phs in entries for p in phs})
phones = ["<blank>"] + phones  # CTC blank at index 0
phone2idx = {p: i for i, p in enumerate(phones)}
```

**Why `<blank>` at index 0?**
CTC requires a special blank token to represent "no phoneme" at a timestep. It's crucial for handling:
- Repeated phonemes: "BUTTER" → "B AH T ER" (not "B AH T T ER")
- Variable-length alignment

---

### 3. **Dataset Class (`G2PCTCDataset`)**

**Purpose:** Convert raw text data into PyTorch tensors.

```python
class G2PCTCDataset(Dataset):
    def __init__(self, entries, char2idx, phone2idx):
        self.data = []
        for word, ph_seq in entries:
            x = [char2idx[c] for c in word]        # Characters → indices
            y = [phone2idx[p] for p in ph_seq]     # Phonemes → indices
            self.data.append((torch.tensor(x), torch.tensor(y)))
```

**Why PyTorch Datasets?**
- Efficient batching
- Automatic shuffling
- Multi-threaded data loading
- Standardized interface

---

### 4. **Batch Collation (`collate_batch` function)**

**The Problem:** Words have different lengths. Neural networks need fixed-size inputs.

**The Solution:**
```python
def collate_batch(batch):
    x_seqs, y_seqs = zip(*batch)
    x_lens = torch.tensor([len(x) for x in x_seqs])
    y_lens = torch.tensor([len(y) for y in y_seqs])
    x_pad = pad_sequence(x_seqs, batch_first=False, padding_value=char2idx["<pad>"])
    y_cat = torch.cat(y_seqs)  # Concatenate for CTC loss
    return x_pad, x_lens, y_cat, y_lens
```

**Why this matters:**
- `pad_sequence`: Adds `<pad>` tokens to make all sequences same length
- `x_lens`: Tells the model which parts are real (not padding)
- `y_cat`: CTC loss expects concatenated targets, not padded sequences
- `batch_first=False`: LSTM expects `(seq_len, batch, features)` format

---

### 5. **Data Splitting**

```python
random.seed(42)
random.shuffle(entries)
train_entries = entries[: int(0.85 * n)]      # 85% training
dev_entries = entries[int(0.85 * n) : int(0.93 * n)]  # 8% validation
test_entries = entries[int(0.93 * n) :]       # 7% testing
```

**Why three splits?**
- **Train**: Model learns from this
- **Dev/Validation**: Tune hyperparameters, monitor overfitting
- **Test**: Final evaluation (never seen during development)

**Why seed 42?** Reproducibility. Same split every run.

---

### 6. **Model Architecture (`LSTMCTCModel`)**

```python
class LSTMCTCModel(nn.Module):
    def __init__(self, n_chars, embed_dim, hidden_dim, n_phones, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(n_chars, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=False, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_phones)  # ×2 for bidirectional
```

**Layer-by-Layer Breakdown:**

1. **Embedding Layer:**
   - Converts character indices to dense vectors
   - `embed_dim=64`: Each character becomes a 64-dimensional vector
   - Why? Characters are discrete symbols; embeddings capture similarity (e.g., vowels cluster together)

2. **LSTM Layer:**
   - `hidden_dim=256`: LSTM memory size
   - `num_layers=2`: Stack two LSTMs for more capacity
   - `bidirectional=True`: Read left-to-right AND right-to-left
   - `dropout=0.1`: Randomly zero 10% of connections to prevent overfitting

3. **Linear (Fully Connected) Layer:**
   - Maps LSTM outputs to phoneme vocabulary
   - `hidden_dim * 2`: Bidirectional doubles the output size
   - Output size = number of phonemes (including blank)

---

### 7. **Forward Pass**

```python
def forward(self, x, lengths):
    emb = self.embed(x)                    # (T,B,E)
    packed = pack_padded_sequence(emb, lengths.cpu(), enforce_sorted=False)
    out, _ = self.lstm(packed)
    out, _ = torch.nn.utils.rnn.pad_packed_sequence(out)
    logits = self.fc(out)                  # (T,B,num_phones)
    log_probs = torch.log_softmax(logits, dim=2)
    return log_probs
```

**Why pack/unpack sequences?**
- `pack_padded_sequence`: Skip computation on padding tokens (efficiency)
- LSTM only processes real data
- `pad_packed_sequence`: Restore padded format for next layer

**Why log_softmax?**
CTC loss expects log probabilities, not raw scores. Softmax converts to probabilities; log keeps numerical stability.

---

### 8. **CTC Loss**

```python
criterion = nn.CTCLoss(blank=phone2idx["<blank>"], zero_infinity=True)
```

**What CTC does:**
1. Considers all possible alignments between input and output
2. Sums probabilities of alignments that produce the target sequence
3. Backpropagates to maximize this sum

**Example:**
- Input: "CAT" (3 characters)
- Output: "K AE T" (3 phonemes)
- Possible alignments: `C→K, A→AE, T→T` or `C→blank, C→K, A→AE, T→T`, etc.

**`zero_infinity=True`:** Prevents crashes when loss becomes infinite (numerical stability).

---

### 9. **Training Loop**

```python
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, xlens, yb, ylens in train_loader:
        optimizer.zero_grad()              # Reset gradients
        log_probs = model(xb, xlens)       # Forward pass
        loss = criterion(log_probs, yb, xlens, ylens)  # Compute loss
        loss.backward()                    # Backpropagation
        optimizer.step()                   # Update weights
```

**Key concepts:**
- **Epoch**: One pass through entire dataset
- **Batch**: Subset of data processed together (parallelization)
- **Gradient descent**: Iteratively adjusting weights to minimize loss

---

### 10. **CTC Decoding (`greedy_decode_ctc`)**

**The Problem:** Model outputs probabilities at each timestep. Need to convert to phoneme sequence.

```python
def greedy_decode_ctc(log_probs):
    indices = log_probs.argmax(dim=2).cpu().numpy().T  # Best phoneme at each step
    preds = []
    for seq in indices:
        prev = None
        out = []
        for i in seq:
            p = idx2phone[i]
            if p == "<blank>":           # Skip blanks
                prev = p
                continue
            if p != prev:                # Collapse repeats
                out.append(p)
            prev = p
        preds.append(out)
    return preds
```

**Greedy vs. Beam Search:**
- **Greedy** (used here): Take most likely phoneme at each step. Fast, but may miss better global solutions.
- **Beam Search**: Consider multiple hypotheses. More accurate but slower.

---

### 11. **Evaluation Metrics**

**Phoneme Error Rate (PER):**
```python
def edit_distance(a, b):
    # Levenshtein distance: min edits to transform a→b
    # Insertions, deletions, substitutions all count as 1 error
```

**PER Calculation:**
```
PER = (Sum of edit distances) / (Total reference phonemes)
```

**Why PER?**
- Standard metric for G2P systems
- Directly measures pronunciation accuracy
- Accounts for insertions, deletions, substitutions

**Example:**
- Predicted: `F IH Z IH K S`
- Reference: `F IH Z IH K S`
- Edit distance: 0 → PER = 0%

---

### 12. **Vocabulary Files**

**Why save vocabularies?**
At inference time, you need the same character→index and phoneme→index mappings used during training.

**`char_vocab_LSTM.json`:**
```json
{"<pad>": 0, "<s>": 1, "</s>": 2, "'": 3, "a": 6, "b": 7, ...}
```

**`phone_vocab_LSTM.json`:**
```json
{"<blank>": 0, "AA": 1, "AE": 2, "AH": 3, ...}
```

**Why two sets of vocab files?**
- `*_LSTM.json`: For CTC model (no stress markers, includes `<blank>`)
- `*.json`: Original format (with stress markers) for other potential uses

---

### 13. **Phoneme Type Reference (`cmudict.phones`)**

```
AA	vowel
AE	vowel
B	stop
CH	affricate
...
```

**Purpose:** Linguistic metadata. Not used in current model but useful for:
- Feature engineering (grouping similar phonemes)
- Error analysis (are vowel errors more common?)
- Future improvements (phoneme embeddings based on type)

---

## Installation & Setup

### Requirements
```bash
pip install torch numpy matplotlib tqdm
```

### Data Preparation
1. Download CMUdict:
   ```bash
   wget http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict.dict
   mv cmudict.dict cmudict.dict.txt
   ```

2. Ensure files are in project directory:
   ```
   ├── cmudict.dict.txt
   ├── LSTMtesting.ipynb
   └── (vocabulary files will be generated)
   ```

---

## Usage

### Training

Open `LSTMtesting.ipynb` and run all cells:

```python
# Key hyperparameters (adjust these in notebook)
BATCH_SIZE = 32          # Batch size
EMBED_DIM = 64           # Character embedding dimension
HIDDEN_DIM = 256         # LSTM hidden size
EPOCHS = 10              # Number of training epochs
LR = 1e-3                # Learning rate
```

**Training will:**
1. Load and preprocess CMUdict
2. Build vocabularies
3. Train the model for 10 epochs (~10-30 minutes on GPU)
4. Save checkpoint to `lstm_ctc_g2p.pth`
5. Plot training loss and validation PER

### Inference

```python
def predict_g2p(word: str):
    seq = torch.tensor([[char2idx.get(c, 0) for c in word.lower()]], dtype=torch.long).T
    length = torch.tensor([seq.size(0)], dtype=torch.long)
    with torch.no_grad():
        log_probs = model(seq.to(DEVICE), length)
    decoded = greedy_decode_ctc(log_probs)[0]
    return decoded

# Test
print(predict_g2p("hello"))  # → H EH L OW
```

---

## Results & Performance

### Typical Performance
- **Training Loss**: Decreases from ~0.40 to ~0.12 over 10 epochs
- **Validation PER**: ~7-8% (model predicts correctly 92-93% of phonemes)
- **Training Time**: ~2-3 minutes/epoch on GPU, ~15-20 minutes on CPU

### Sample Predictions

| Word | Predicted | Reference | PER |
|------|-----------|-----------|-----|
| google | G UW G AH L | G UW G AH L | 0% |
| physics | F IH Z IH K S | F IH Z IH K S | 0% |
| algorithm | AE L G ER IH DH AH M | AE L G ER IH DH AH M | 0% |
| data | D EY T AH | D AE T AH | 25% |

**Note:** "data" has multiple valid pronunciations; model learned one variant.

---

## Technical Deep Dive

### Why Bidirectional LSTM?

**Problem:** English pronunciation depends on context:
- "th" in "the" vs "think"
- Silent 'e' in "make" (affects previous vowel)
- "ph" → "F" sound

**Solution:** Read the word twice:
1. **Forward** (left→right): "k-n-i-g-h-t"
2. **Backward** (right←left): "t-h-g-i-n-k"

Model combines both views to understand full context.

### Why CTC Loss?

**Alignment Problem Example:**
```
Word:     "CAT"        (3 characters)
Phonemes: "K AE T"     (3 phonemes)
```

Simple mapping: C→K, A→AE, T→T works here.

But:
```
Word:     "KNIGHT"     (6 characters)
Phonemes: "N AY T"     (3 phonemes)
```

Which characters map to which phonemes? K is silent, GH together make nothing, I and GH combine for AY sound.

**CTC automatically learns these complex alignments.**

### Dropout Regularization

```python
dropout=0.1
```

**Purpose:** Prevent overfitting (memorizing training data).

**How it works:**
- Randomly "drop" 10% of LSTM connections during training
- Forces model to learn robust features
- Turned off during evaluation

### Packing/Unpacking Sequences

**Why pack?**
```python
packed = pack_padded_sequence(emb, lengths.cpu(), enforce_sorted=False)
```

**Efficiency:** LSTM only processes real data, skips padding.

**Example:**
```
Batch of 3 words: ["cat", "hello", "a"]
Padded:   ["cat  ", "hello", "a    "]
Packed:   Process only: c,a,t,h,e,l,l,o,a (9 steps instead of 18)
```

---

## Synthesizer Demo

### Overview

The ultimate goal of this project is to have a fully integrated text-to-speech model. The workflow operates as follows:

**Text Input → LSTM Model Prediction → ARPAbet to SAMPA Translation → MBROLA .wav Output**

A Hugging Face Docker repository currently houses this project at:  
https://huggingface.co/spaces/TalkNetTeamMA416/TalkingModernNETtalkTeam

However, it must be run locally as it does not build on the website. Follow the steps below to run this project locally with audio support.

---

### Prerequisites

Before starting, ensure you have the following installed and configured:

- **Windows 10/11** (or compatible OS)
- **WSL2** enabled (for Windows users)
- **Docker Desktop** installed and configured to use the WSL2 backend
- **Hugging Face account** and access token  
  Create a token here: https://huggingface.co/settings/tokens

---

### Installation and Setup

#### Step 1: Authenticate with the Hugging Face Docker Registry

Open PowerShell or your terminal and run:
```bash
docker login registry.hf.space
```

When prompted, enter:
- **Username**: Your Hugging Face username
- **Password**: Your Hugging Face access token

A successful login will display:
```
Login Succeeded
```

#### Step 2: Pull the Latest Container Image

The Space automatically builds and publishes a fresh container whenever updates are pushed. To pull the latest version, run:
```bash
docker pull registry.hf.space/talknetteamma416-talkingmodernnettalkteam:latest
```

**Note**: This is a lengthy download (~3.5 GB). If the build recently completed on Hugging Face, this command will fetch the newest image.

#### Step 3: Run the Container Locally

Gradio defaults to binding only to `127.0.0.1` inside the container, which prevents external access. To fix this, use the environment variable `GRADIO_SERVER_NAME=0.0.0.0`.

Run the following command:
```bash
docker run -it --rm \
  -p 7860:7860 \
  --platform=linux/amd64 \
  -e GRADIO_SERVER_NAME=0.0.0.0 \
  registry.hf.space/talknetteamma416-talkingmodernnettalkteam:latest
```

**For PowerShell users**, use backticks for line continuation:
```powershell
docker run -it --rm `
  -p 7860:7860 `
  --platform=linux/amd64 `
  -e GRADIO_SERVER_NAME=0.0.0.0 `
  registry.hf.space/talknetteamma416-talkingmodernnettalkteam:latest
```

You should see output indicating:
```
Running on local URL: http://0.0.0.0:7860
```

This means the service is accessible from outside the container.

#### Step 4: Open the Demo in Your Browser

Navigate to one of the following URLs:

- **Primary**: http://localhost:7860
- **Alternative**: http://127.0.0.1:7860

If neither works, you may need to use the WSL2 network IP. Find it by running:
```bash
wsl hostname -I
```

Then visit:
```
http://<WSL-IP>:7860
```

---

### Using the Synthesizer

Once the web interface loads:

1. Enter your text into the input textbox
2. Click **Submit**
3. After processing, a visualized display of your audio will appear on the right
4. You can play, speed up or slow down, scrub through, and download the generated audio

**Test your skills!** Try the provided phrases and examples on the HTML page titled **ModelDemoTestExamples.html**. See if you can score 5/5 on the provided movie quotes!

---

### Stopping the Container

To stop the container:

- **If running interactively**: Press `Ctrl + C`
- **Manual stop**:
```bash
  docker ps
  docker stop <container_id>
```

Containers started with the `--rm` flag automatically clean themselves up when stopped.

---

### Troubleshooting

- **Download too slow?** The image is approximately 3.5 GB, so ensure you have a stable internet connection.
- **Port already in use?** If port 7860 is occupied, change the port mapping to `-p 8080:7860` (or another available port) and access via `http://localhost:8080`.
- **WSL2 networking issues?** Ensure Docker Desktop is properly configured to use the WSL2 backend in Settings → Resources → WSL Integration.

---

## Advanced Topics

### Beam Search Decoding

Current implementation uses greedy decoding (fast but suboptimal). For better accuracy:

```python
# Pseudocode
def beam_search(log_probs, beam_width=5):
    # Keep top-k hypotheses at each step
    # Return best overall sequence
```

### Transfer Learning

Pre-trained embeddings could improve performance:
```python
# Use word embeddings (GloVe, FastText)
pretrained_embeddings = load_pretrained()
model.embed.weight.data.copy_(pretrained_embeddings)
```

### Attention Mechanism

Adding attention would let the model explicitly learn character-phoneme alignments:
```python
class LSTMWithAttention(nn.Module):
    # Add attention layer between LSTM and output
```

---

## Troubleshooting

### Low Accuracy
- Increase `HIDDEN_DIM` (more capacity)
- Add more `num_layers`
- Train longer (`EPOCHS`)
- Reduce `dropout` if underfitting

### Overfitting
- Increase `dropout`
- Reduce model size
- Get more data
- Add data augmentation

### Out of Memory
- Reduce `BATCH_SIZE`
- Reduce `HIDDEN_DIM`
- Use gradient accumulation

---

## Future Improvements

1. **Stress Prediction**: Restore stress markers for more accurate pronunciation
2. **Multi-Head Attention**: Replace/augment LSTM with Transformers
3. **Morphological Features**: Incorporate word structure (prefixes, suffixes)
4. **Multilingual**: Extend to other languages
5. **End-to-End TTS**: Combine with audio generation (Tacotron, WaveNet)

---

## References

- [CMUdict Documentation](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
- [CTC Loss Paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf) (Graves et al., 2006)
- [Sequence Modeling with CTC](https://distill.pub/2017/ctc/)
- [PyTorch CTC Loss](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)

---

## License

This project uses CMUdict, which is in the public domain.

## Author

Developed for phonetic modeling and G2P research.