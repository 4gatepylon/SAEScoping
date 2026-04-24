import os, litellm
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
try:
    resp = litellm.completion(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    print(f"Success: {resp.choices[0].message.content}")
except Exception as e: print(f"Error: {e}")
