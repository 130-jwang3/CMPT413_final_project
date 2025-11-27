from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from BigModel.Base import BigModelBase

class Chat_AzureGPT4o(BigModelBase):
    def __init__(self, max_tokens=1024, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.tokens_per_minute = 50000
        self.credential = AzureCliCredential()
        self.token_provider = get_bearer_token_provider(
            self.credential,
            "https://cognitiveservices.azure.com/.default"
        )
        self.client = AzureOpenAI(
            azure_endpoint="https://ai-wangjacky0555814ai832936423341.openai.azure.com/",
            azure_ad_token_provider=self.token_provider,
            api_version="2025-01-01-preview",
            max_retries=5,
        )
        self.used_token = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }
    def chat(self, message:list):
        # clear last used token
        self.last_used_token['completion_tokens'] = 0
        self.last_used_token['prompt_tokens'] = 0
        self.last_used_token['total_tokens'] = 0
        # chat
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=message,
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        ans = response.choices[0].message.content
        # token usage
        self.last_used_token['completion_tokens'] = response.usage.completion_tokens
        self.last_used_token['prompt_tokens'] = response.usage.prompt_tokens
        self.last_used_token['total_tokens'] = response.usage.total_tokens
        # # token count
        self.used_token['completion_tokens'] += response.usage.completion_tokens
        self.used_token['prompt_tokens'] += response.usage.prompt_tokens
        self.used_token['total_tokens'] += response.usage.total_tokens

        return ans

    def text_item(self, content):
        item = {
            "type": "text",
            "text": content
        }
        return item

    def image_item_from_path(self, image_path, detail="high"):
        image_base64 = self.image_encoder_base64(image_path)
        item = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}", 
                "detail": detail,
            },
        }
        return item
    
    def image_item_from_base64(self, image_base64, detail="high"):
        item = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}", 
                "detail": detail,
            },
        }
        return item

class Chat_AzureGPT4o_mini(BigModelBase):
    def __init__(self, max_tokens=1024, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.tokens_per_minute = 50000
        self.credential = AzureCliCredential()
        self.token_provider = get_bearer_token_provider(
            self.credential,
            "https://cognitiveservices.azure.com/.default"
        )
        self.client = AzureOpenAI(
            azure_endpoint="https://msraopenaieastus.openai.azure.com/",
            azure_ad_token_provider=self.token_provider,
            api_version="2024-07-01-preview",
            max_retries=5,
        )
        self.used_token = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }
    def chat(self, message:list):
        # clear last used token
        self.last_used_token['completion_tokens'] = 0
        self.last_used_token['prompt_tokens'] = 0
        self.last_used_token['total_tokens'] = 0
        # chat
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message,
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        ans = response.choices[0].message.content
        # token usage
        self.last_used_token['completion_tokens'] = response.usage.completion_tokens
        self.last_used_token['prompt_tokens'] = response.usage.prompt_tokens
        self.last_used_token['total_tokens'] = response.usage.total_tokens
        # # token count
        self.used_token['completion_tokens'] += response.usage.completion_tokens
        self.used_token['prompt_tokens'] += response.usage.prompt_tokens
        self.used_token['total_tokens'] += response.usage.total_tokens

        return ans
    
    def text_item(self, content):
        item = {
            "type": "text",
            "text": content
        }
        return item

    def image_item_from_path(self, image_path, detail="high"):
        image_base64 = self.image_encoder_base64(image_path)
        item = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}", 
                "detail": detail,
            },
        }
        return item
    
    def image_item_from_base64(self, image_base64, detail="high"):
        item = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}", 
                "detail": detail,
            },
        }
        return item 

class Chat_DeepSeekR1(BigModelBase):
    def __init__(self, max_tokens=1024, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.tokens_per_minute = 50000
        self.credential = AzureCliCredential()
        self.token_provider = get_bearer_token_provider(
            self.credential,
            "https://cognitiveservices.azure.com/.default"
        )
        self.client = AzureOpenAI(
            azure_endpoint="https://ai-lwa1180883ai153989469943.openai.azure.com",
            azure_ad_token_provider=self.token_provider,
            api_version="2024-05-01-preview",
            max_retries=5,
        )
        self.used_token = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }
    def chat(self, message:list):
        # clear last used token
        self.last_used_token['completion_tokens'] = 0
        self.last_used_token['prompt_tokens'] = 0
        self.last_used_token['total_tokens'] = 0
        # chat
        response = self.client.chat.completions.create(
            model="DeepSeek-R1",
            messages=message,
            # response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        ans = response.choices[0].message.content
        
        # token usage
        self.last_used_token['completion_tokens'] = response.usage.completion_tokens
        self.last_used_token['prompt_tokens'] = response.usage.prompt_tokens
        self.last_used_token['total_tokens'] = response.usage.total_tokens
        # # token count
        self.used_token['completion_tokens'] += response.usage.completion_tokens
        self.used_token['prompt_tokens'] += response.usage.prompt_tokens
        self.used_token['total_tokens'] += response.usage.total_tokens

        try:
            start = ans.index('```json') + 7  # Skip ```json
            end = ans.rindex('```')           # Before closing ```
            json_str = ans[start:end].strip()
            print(f"Extracted JSON: {repr(json_str)}")  # Debug
            return json_str
        except ValueError:
            print(f"Could not extract JSON from: {repr(ans)}")
            return ans  # Fallback, handle in chat_with_rate_control
    
    def text_item(self, content):
        item = {
            "type": "text",
            "text": content
        }
        return item

    def image_item_from_path(self, image_path, detail="high"):
        image_base64 = self.image_encoder_base64(image_path)
        item = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}", 
                "detail": detail,
            },
        }
        return item
    
    def image_item_from_base64(self, image_base64, detail="high"):
        item = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}", 
                "detail": detail,
            },
        }
        return item 