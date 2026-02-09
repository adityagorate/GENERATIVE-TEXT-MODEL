from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Function to generate text
def generate_text(prompt, max_length=120):
    inputs = tokenizer.encode(prompt, return_tensors="pt")     # Convert text to tokens

    # Generate text

    output = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2
    )

    # Convert tokens back to text

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Main program

if __name__ == "__main__":
    topic = input("Enter topic: ")
    text = generate_text(topic)
    print("\nGenerated Text:\n")
    print(text)
