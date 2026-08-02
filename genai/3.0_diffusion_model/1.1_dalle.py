"""DALL-E image generation exercises."""

# !pip install openai

from openai import OpenAI

# https://platform.openai.com/docs/models for a list of models
# Read OPENAI_API_KEY from the process environment; never put it in source.
client = OpenAI()

# generte image
response = client.images.generate(
    model="dall-e-2",
    prompt="an astronaut in a space suit riding a unicorn",
    size="512x512",
    quality="standard",
    n=1,
)
image_url = response.data[0].url
print(image_url)

# exercise 15.2
response = client.images.generate(
    model="dall-e-2",
    prompt="a cat in a suit working on a computer",
    size="512x512",
    quality="standard",
    n=1,
)
image_url = response.data[0].url
print(image_url)
