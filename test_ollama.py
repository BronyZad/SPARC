import json
import requests

URL = "http://localhost:11434/api/chat"
MODEL = "gpt-oss:20b"

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Why is the sky blue? Answer in exactly one sentence."}
    ],
    "options": {
        "temperature": 0.1,
        "num_predict": 150
    },
    "stream": True  # 🟢 Force streaming over the network
}

print(f"📡 Sending raw HTTP POST request to {URL}...")

try:
    # Send request with streaming enabled
    response = requests.post(URL, json=payload, stream=True)
    response.raise_for_status()

    print("🔍 Raw Token Stream:")
    print("-" * 50)
    
    full_output = ""
    for line in response.iter_lines():
        if line:
            # Ollama returns each chunk as a separate JSON object string
            chunk = json.loads(line.decode('utf-8'))
            token = chunk.get("message", {}).get("content", "")
            
            if token:
                print(token, end="", flush=True)
                full_output += token

    print("\n" + "-" * 50)
    print(f"✅ Finished. Total captured length: {len(full_output)} characters.")

except Exception as e:
    print(f"\n❌ HTTP Connection Failed: {e}")