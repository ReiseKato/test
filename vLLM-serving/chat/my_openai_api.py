from openai import OpenAI, OpenAIError
import json
import datetime
from config import API_BASE, API_KEY, DEFAULT_MODEL, GENERATION_PARAMS

RESONSE_LOG_PATH = "../responses/responses.jsonl"

def connect_to_openai():
    '''
    Connect to OpenAI API and return the OpenAI client if successful.
    This function is used to establish a connection to the OpenAI API using the provided API key and base URL.
    It has to be called before making any requests to the API.
    It handles any exceptions that may occur during the connection process and logs them appropriately.

    Returns:
        OpenAI client object if the connection is successful, None otherwise.
    Raises:
        OpenAIError: If there is an error connecting to the OpenAI API.
        Exception: For any other unexpected errors.
    '''
    try:
        client = OpenAI(
            base_url=API_BASE,
            api_key=API_KEY,
        )
        print("[INFO] Connected to OpenAI API.")
        if client:
            return client
        return None
    except OpenAIError as e:
        print(f"[ERROR] OpenAI API error: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def ask_chat(messages, model, generation_params=None):
    '''
    Send a chat request to the OpenAI API using the provided messages and model. It returns the response as an OpenAI object.
    It handles any exceptions that may occur during the request process and logs them appropriately.
    
    Args:
        messages (list): A list of messages to send to the OpenAI API.
        model (str): The model to use for the chat request.
        generation_params (dict, optional): Additional parameters for the chat request. Defaults to None.
    
    Returns:
        object: The response from the OpenAI API.
    
    Raises:
        OpenAIError: If there is an error with the OpenAI API request.
        Exception: For any other unexpected errors.
    '''
    try:
        client = connect_to_openai()
        if not client:
            return None
 
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **(generation_params or GENERATION_PARAMS)
        )

        log_response(messages, response, model)
        return response

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def get_summary(messages, model, generation_params=None):
    '''
    Send a chat request to the OpenAI API and return only the generated summary content.

    Args:
        messages (list): A list of messages to send to the OpenAI API.
        model (str): The model to use for the chat request.
        generation_params (dict, optional): Additional parameters for the chat request. Defaults to None.

    Returns:
        str: The generated summary content, or None if the request fails.
    '''
    try:
        client = connect_to_openai()
        if not client:
            return None

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **(generation_params or GENERATION_PARAMS)
        )

        # Extract and return the content of the assistant's message
        return response.choices[0].message.content if response else None

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return None

def log_response(messages, response, model):
    '''
    Log the response from the OpenAI API to a JSONL file.
    This function is used to save the response data, including the model used, timestamp, request messages, and response content.
    It appends the log data to a file named "responses.jsonl" in JSON format.
    
    Args:
        messages (list): A list of messages sent to the OpenAI API.
        response (object): The response object from the OpenAI API.
        model (str): The model used for the chat request.
    
    Raises:
        Exception: For any unexpected errors during the logging process.
    '''
    log_data = {
        "model": model,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "request": messages,
        "response": response.model_dump()
    }
    with open(RESONSE_LOG_PATH, "a") as f:
        f.write(json.dumps(log_data) + "\n")

def format_chat_prompt(user_input):
    '''
    Format the user input into a chat prompt for the OpenAI API.
    This function creates a list of messages with a system message and the user input.

    Args:
        user_input (str): The user input to format.
    
    Returns:
        list: A list of messages formatted for the OpenAI API.
    '''
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input},
    ]

def list_models():
    '''
    List available models from the OpenAI API.

    This function retrieves the list of models from the OpenAI API and prints their details (model id, raw mdel name, context size).

    Returns:
        list: A list of available model IDs.
    
    Raises:
        Exception: For any unexpected errors during the model listing process.
    '''
    try:
        client = connect_to_openai()
        if not client:
            return

        models = client.models.list()
        model_ids = []
        print(f"[INFO] Available models: \n")
        for i, model in enumerate(models.data):
            print(f"{i} - {model.id} - {model.root} - Max model len: {model.max_model_len}\n")
            model_ids.append(model.id)
        
        return model_ids
            
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def choose_model():
    '''
    Choose a model from the list of available models.
    It retrieves the list of models using the `list_models` function and prompts the user to select one by entering its corresponding number.
    If the user enters an invalid choice, it will prompt again until a valid choice is made.
    
    Returns:
        str: The selected model ID. If no models are available, it will return None.
    
    Raises:
        Exception: For any unexpected errors during the model selection process.
    '''
    try:
        model_ids = list_models()
        print(f"model_ids: {model_ids}")
        if not model_ids:
            print("[ERROR] No models available.")
            return
        
        while True:
            choice = input("\nEnter the number of the model you want to use: ").strip()
            if choice.isdigit() and 0 <= int(choice) <= len(model_ids) - 1:
                selected_model = model_ids[int(choice)]
                print(f"[INFO] Selected model: {selected_model}")
                return selected_model
            else:
                print("[ERROR] Invalid choice. Please enter a valid number.")
    except Exception as e:
        print(f"[ERROR] Unexpected error while choosing model: {e}")
        return None