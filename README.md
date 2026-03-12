# Energy QA Model Fine-Tuning

This project fine-tunes a GPT-2 language model on energy-related question-answering data
and provides a GUI to compare the original and fine-tuned model responses side by side.

## Project Structure

| File | Purpose |
|---|---|
| `fine_tune.py` | Step 3 – Fine-tune GPT-2 on `data/energy_qa.txt` |
| `test_model.py` | Step 4 – Test original vs fine-tuned model generation |
| `gui.py` | Step 5 – Tkinter GUI for side-by-side comparison |
| `config.py` | Shared configuration (model name, paths, hyperparameters) |
| `data/energy_qa.txt` | 50 energy Q&A pairs used as training data |

## Project Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hmohamed080/energy-qa-finetuning.git
   cd energy-qa-finetuning
   ```

2. **Install dependencies** (Python 3.8+ required):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 3 – Fine-Tune the Model

```bash
python fine_tune.py
```

The script loads `data/energy_qa.txt`, fine-tunes GPT-2 for 3 epochs, and saves the
result to `./fine_tuned_model/`.

### Step 4 – Test the Model

```bash
python -m unittest test_model.py -v
```

Tests confirm that both the original and fine-tuned models generate text and meet
basic response-time requirements.

### Step 5 – Launch the GUI

```bash
python gui.py
```

A window opens with a question input field.  Type any energy-related question and
click **Compare Models** to see the original GPT-2 response and the fine-tuned
response displayed side by side.

## Additional Information

For report guidance, see `REPORT_TEMPLATE.md`.
