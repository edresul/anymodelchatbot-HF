from transformers import pipeline

# Replace with your Hugging Face token
HUGGINGFACE_TOKEN = ""  # <-- Add your token here

# Load the model with authentication
chatbot = pipeline(
    task="text-generation",
    model="",  # <--- put the correct model name / model URL here without the domain. example: Qwen/Qwen3-Next-80B-A3B-Instruct instead of https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
    use_auth_token=HUGGINGFACE_TOKEN,  # <-- Pass your token here
    device=-1  # Use GPU (0) or CPU (-1)
)

def generate_response(prompt, max_length=50, temperature=0.7):
    response = chatbot(
        prompt,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        pad_token_id=chatbot.tokenizer.eos_token_id
    )
    return response[0]["generated_text"]

def chat():
    """
    Main chat loop to interact with the model.
    """
    print("Welcome to the AnyModel Chatbot For HuggingFace! Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Generate the bot's response
        bot_reply = generate_response(user_input)
        print(f"Bot: {bot_reply}")

if __name__ == "__main__":
    chat()