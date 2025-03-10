# Assume openai>=1.0.0
from openai import OpenAI
from PIL import Image
import base64
# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key="HwFKgryu2LwqFv6lV1ZlIixWh67NJgvH",
    base_url="https://api.deepinfra.com/v1/openai",
)


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "samples\gender\store_image.jpg"

# Getting the Base64 string
base64_image = encode_image(image_path)

chat_completion = openai.chat.completions.create(
    # model="meta-llama/Llama-3.2-11B-Vision-Instruct",
    model="meta-llama/Llama-3.2-90B-Vision-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze this retail store image for gender demographics. How many men and women do you see in the image and what products are they looking at?

                            Please return your analysis as a JSON object with the following keys:
                            - mencount: number of men in the image
                            - womencount: number of women in the image
                            - products: list of products they are looking at
                            - insights: your analysis of what this means for the store

                            Example format:
                            {
                            "mencount": 2,
                            "womencount": 3,
                            "products": ["clothing", "electronics", "accessories"],
                            "insights": "More women than men suggesting a female-oriented shopping experience."
                            }""",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
    max_tokens=32000,
)

print(chat_completion.choices[0].message.content)
print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)

# Hello! It's nice to meet you. Is there something I can help you with, or would you like to chat?
# 11 25
