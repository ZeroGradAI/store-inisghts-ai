# Assume openai>=1.0.0
from openai import OpenAI

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key="HwFKgryu2LwqFv6lV1ZlIixWh67NJgvH",
    base_url="https://api.deepinfra.com/v1/openai",
)

chat_completion = openai.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
)

print(chat_completion.choices[0].message.content)
print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)

# Hello! It's nice to meet you. Is there something I can help you with, or would you like to chat?
# 11 25
