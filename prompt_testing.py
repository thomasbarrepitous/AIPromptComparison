from typing import List, Dict, Any
from abc import ABC, abstractmethod
import os
import asyncio
from dotenv import load_dotenv

class AIModel(ABC):
    """Abstract base class for AI models"""
    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        pass

class OpenAIModel(AIModel):
    """OpenAI implementation"""
    def __init__(self, api_key: str):
        import json
        from openai import OpenAI
        with open('configs/openai_config.json') as config_file:
            config = json.load(config_file)
            self.client = OpenAI(
                api_key=api_key
            )
            self.model = config.get('model')
            self.kwargs = config.get('kwargs')

    async def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self.kwargs
        )
        return response.choices[0].message.content

class GeminiModel(AIModel):
    """Google's Gemini"""
    def __init__(self, api_key: str):
        import json
        with open('configs/gemini_config.json') as config_file:
            config = json.load(config_file)
            self.model = config.get('model')
            self.kwargs = config.get('kwargs')
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name=self.model)

    async def generate_response(self, prompt: str) -> str:
        response = self.model.start_chat(
            history=[],
            **self.kwargs
        )
        response = response.send_message(prompt)
        return response.text

class DeepseekModel(AIModel):
    """Deepseek API implementation"""
    def __init__(self, api_key: str):
        import json
        from openai import OpenAI
        with open('configs/deepseek_config.json') as config_file:
            config = json.load(config_file)
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"  # DeepSeek's API endpoint
            )
            self.model = config.get('model')
            self.kwargs = config.get('kwargs')

    async def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            **self.kwargs
        )
        return response.choices[0].message.content

class MistralModel(AIModel):
    """Mistral API implementation"""
    def __init__(self, api_key: str):
        import json
        from openai import OpenAI
        with open('configs/mistral_config.json') as config_file:
            config = json.load(config_file)
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.mistral.ai/v1"  # Mistral's API endpoint
            )
            self.model = config.get('model')
            self.kwargs = config.get('kwargs')

    async def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            **self.kwargs
        )
        return response.choices[0].message.content

class Prompt:
    def __init__(self, url: str):
        self.get_text(url)

    # Get prompt from prompt.txt
    def get_text(self, url: str):
        try:
            with open(url, 'r') as file:
                self.text = file.read()
        except FileNotFoundError:
            print(f"{url} not found")
            self.text = ""

async def main():
    """Main function to run the prompt tester"""
    prompt = Prompt("prompt.txt")
    print(prompt.text)

    # Load environment variables
    load_dotenv()

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    print(os.getenv("OPENAI_API_KEY"))
    print(os.getenv("GEMINI_API_KEY"))

    models = [
        OpenAIModel(
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        GeminiModel(
            api_key=os.getenv("GEMINI_API_KEY")
        ),
        DeepseekModel(
            api_key=os.getenv("DEEPSEEK_API_KEY")
        ),
        MistralModel(
            api_key=os.getenv("MISTRAL_API_KEY")
        )
    ]

    for model in models:
        # Generate responses in independent .txt files
        with open(f"results/{model.__class__.__name__}.txt", "w") as file:
            file.write(await model.generate_response(prompt.text))

if __name__ == "__main__":
    asyncio.run(main())

