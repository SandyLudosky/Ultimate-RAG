from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
MODEL = "gpt-4-turbo"
PROMPT = "Identify and describe in 2 sentences what’s in this image?"

def describe_image(image_path): 
    response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {
            "type": "image_url",
            "image_url": {
                "url": image_path,
            },
            },
        ],
        }
    ],
    max_tokens=300,
    )
    return response.choices[0].message.content
