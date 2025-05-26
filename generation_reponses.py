import requests
import polars as pl

OLLAMA_HOST = "http://localhost:11434"
MODELS_NAMES = [
    "phi4-mini:3.8b", 
    "hf.co/bartowski/Ministral-8B-Instruct-2410-GGUF:Q4_K_M",
    "gemma3:4b",
    "llama3.1:8b",
    "deepseek-r1:8b"
]
MAXIMUM_TOKEN_LENGTH_OUTPUT = 2048

def get_response(model_name, prompt):
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": 0,
            "options": {
                "num_predict": MAXIMUM_TOKEN_LENGTH_OUTPUT,
                "temperature": 0.5
            }
        }
    )
    response_value = response.json()["response"] if (response.ok) else f"Erreur : {response.status_code} {response.text}"

    return response_value

def create_response_file(df, model_name):
    response = [get_response(model_name, prompt) for prompt in df["prompt"]]
    new_df = df.with_columns([
        pl.lit(model_name).alias("model_name"),
        pl.Series("response", response)
    ])

    new_df.write_csv(f"data/results/results_{model_name[0]}.csv")

df = pl.read_csv("data/final_data.csv")

for model_name in MODELS_NAMES:
    create_response_file(df, model_name)
