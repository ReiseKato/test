import os
import json
from util import evaluate_summaries

def evaluate_and_save_results(input_path, output_dir):
    """
    Evaluate the summaries in the given JSON file and save the results.
    :param input_path: Path to the input JSON file containing the dataset.
    :param output_dir: Directory to save the evaluation results.
    """
    # Load the dataset
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset_name = dataset["dataset_name"]
    model_name = os.path.basename(os.path.dirname(input_path))
    data = dataset["data"]

    # Extract references and generated summaries
    references = [entry["reference"] for entry in data]
    generated_summaries = [entry["summary"] for entry in data]

    # Evaluate summaries
    print("[INFO] Evaluating summaries...")
    evaluation_results = evaluate_summaries(references, generated_summaries)

    # Prepare output directory
    output_dir = os.path.join(output_dir, dataset_name, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Determine the new file name based on the number of existing files
    existing_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
    new_file_index = len(existing_files)
    output_path = os.path.join(output_dir, f"{new_file_index}.json")

    # Save the evaluation results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)

    print(f"[INFO] Evaluation results saved to {output_path}")


if __name__ == "__main__":
    # Input JSON file path
    input_path = os.path.join(os.path.dirname(__file__), "../chat/summaries/mlsum_de/phi/0.json")

    # Output directory for evaluation results
    output_dir = os.path.join(os.path.dirname(__file__), "../evaluation")

    # Evaluate and save results
    evaluate_and_save_results(input_path, output_dir)