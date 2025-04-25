from my_openai_api import *
from util import *

def chat():
    """
    Main function to run the chat application.
    It handles user input, model selection, and chat interactions.
    The user can choose a model and start a chat session.
    The chat continues until the user types 'exit' or 'quit'.
    """
    # Select model
    model = choose_model()
    if not model:
        print("[ERROR] No model selected. Exiting.")
        exit(1)
    
    # Chat loop
    print(f"[INFO] Starting chat with model: {model}")
    print("Type 'exit' or 'quit' to end the chat.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        messages = format_chat_prompt(user_input)
        response = ask_chat(messages, model)
        if response:
            content = response.choices[0].message.content
            print(f"Assistant: {content}")

def summarize():
    """
    Main function to run the summarization application.
    It handles user input, model selection, and summarization interactions.
    The user can choose a dataset, a model, and start a summarization session.
    The summarization continues until the user types 'exit' or 'quit'.
    """
    # Select model
    model = choose_model()
    if not model:
        print("[ERROR] No model selected. Exiting.")
        exit(1)
    
    # Select dataset
    dataset = choose_dataset()
    if not dataset:
        print("[ERROR] No dataset selected. Exiting.")
        exit(1)

    copied_dataset = {
        "dataset_name": dataset["dataset_name"],
        "data": dataset["data"]
    }

    # Summarization loop
    print(f"[INFO] Starting summarization with model: {model}")
    for entry in copied_dataset["data"]:
        text = entry["text"]
        print(f"[INFO] Generating summary for text: {text[:100]}...")  # Show a snippet of the text

        # Generate summary using the model
        messages = format_chat_prompt(f"Fasse den folgenden Text zusammen:\n\n{text}")
        summary = get_summary(messages, model)
        if summary:
            entry["summary"] = summary
        else:
            print("[ERROR] Failed to generate summary. Leaving the field empty.")
            entry["summary"] = ""

    # Determine the new file name based on the number of existing files    
    output_dir = os.path.join("summaries", copied_dataset["dataset_name"], str(model))
    os.makedirs(output_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
    new_file_index = len(existing_files)
    output_path = os.path.join(output_dir, f"{new_file_index}.json")
    
    # Save the updated dataset to a new file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(copied_dataset, f, ensure_ascii=False, indent=4)

    print(f"[INFO] Updated dataset with summaries saved to {output_path}")


if __name__ == "__main__":
    # save_mlsum_to_json() # save mlsum dataset to json file
    summarize()
    

    