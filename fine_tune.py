import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Fine-tune model function
def fine_tune_model(train_texts, epochs=1, batch_size=2):
    # Tokenize input texts
    input_ids = tokenizer(train_texts, return_tensors='tf', padding=True, truncation=True).input_ids
    dataset = tf.data.Dataset.from_tensor_slices(input_ids).batch(batch_size)
    
    # Define an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    
    # Compile the model
gpt2_model = model
gpt2_model.compile(optimizer=optimizer)
    
    # Fine-tune the model
    gpt2_model.fit(dataset, epochs=epochs)

# Example usage
if __name__ == '__main__':
    texts = ["Your text here..."]  # Replace with your training data
    fine_tune_model(texts, epochs=3)
