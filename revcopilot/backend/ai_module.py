# AI Module for RevCopilot

import openai

class AIAnalyzer:
    def __init__(self, api_key):
        openai.api_key = api_key

    def analyze_code(self, code_snippet):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Explain the following code:\n{code_snippet}"}
            ]
        )
        return response.choices[0].message['content']

    def get_hints(self, code_snippet):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Provide hints for understanding the following code:\n{code_snippet}"}
            ]
        )
        return response.choices[0].message['content']