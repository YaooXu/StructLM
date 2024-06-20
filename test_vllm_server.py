from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "210.75.240.150:12240"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

message_prompt = {"role": "user", "content": 'who are you?'}

completion = client.chat.completions.create(model="meta-llama/Meta-Llama-3-70B-Instruct",
                                      message_prompt=message_prompt)

print(completion)