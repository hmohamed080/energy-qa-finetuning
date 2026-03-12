import os
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

FINE_TUNED_MODEL_DIR = './fine_tuned_model'
DATA_PATH = 'data/energy_qa.txt'


class EnergyQADataset(Dataset):
    """Dataset for energy QA text blocks."""

    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item


def load_energy_texts(file_path=DATA_PATH):
    """Parse Q&A blocks from the energy dataset file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = [block.strip() for block in content.strip().split('\n\n') if block.strip()]
    return blocks


def fine_tune_model(data_path=DATA_PATH, output_dir=FINE_TUNED_MODEL_DIR,
                    epochs=3, batch_size=2, max_length=256):
    """Fine-tune GPT-2 on the energy QA dataset and save the result."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained('gpt2')

    texts = load_energy_texts(data_path)
    encodings = tokenizer(texts, truncation=True, padding='max_length',
                          max_length=max_length, return_tensors='pt')
    dataset = EnergyQADataset(encodings)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")


if __name__ == '__main__':
    fine_tune_model()
