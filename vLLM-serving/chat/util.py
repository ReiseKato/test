import os
import json
from datasets import load_dataset

def save_mlsum_to_json():
    """
    Save the MLSUM dataset to a JSON file with the structure:
    {
        "dataset_name": <name of the dataset>,
        "data": [
            {
                "text": <text>,
                "reference": <summary>,
                "summary": ""
            },
            ...
        ]
    }
    """
    output_path = os.path.join("data", "mlsum_de.json")
    dataset_name = "mlsum_de"
    data = []
    mlsum = load_dataset("mlsum", "de")["test"]
    for text, summary in zip(mlsum["text"][:30], mlsum["summary"][:30]):
        data.append({
            "text": text,
            "reference": summary,
            "summary": ""  # Placeholder for reference summary
        })
    
    output = {
        "dataset_name": dataset_name,
        "data": data
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"[INFO] Dataset saved to {output_path}")

def choose_dataset():
    """
    Choose a dataset for summarization.
    This function lists available datasets and allows the user to select one.
    """
    # List available datasets in the 'data' directory
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory '{data_dir}' does not exist.")
        return

    datasets = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    if not datasets:
        print(f"[ERROR] No datasets found in '{data_dir}'.")
        return

    print("[INFO] Available datasets:")
    for idx, dataset in enumerate(datasets):
        print(f"{idx}. {dataset}")
    
    # Let the user choose a dataset
    try:
        choice = int(input("Choose a dataset by number: ").strip())
        if choice < 0 or choice >= len(datasets):
            print("[ERROR] Invalid choice. Exiting.")
            return
        dataset_path = os.path.join(data_dir, datasets[choice])
    except ValueError:
        print("[ERROR] Invalid input. Exiting.")
        return

    # Load the selected dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print(f"[INFO] Loaded dataset: {dataset['dataset_name']}")
    return dataset