from zhipuai import ZhipuAI

from BigModel.Base import BigModelBase

'''
General GLM-4v
'''
class Chat_GLM4v(BigModelBase):
    def __init__(self, max_tokens=1024, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.load_my_api('chatglm')
        self.client = ZhipuAI(api_key=self.api_key)
        self.model_name = 'glm-4v-plus-0111'
        self.tokens_per_minute = 60000
        self.used_token = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }

    def chat(self, messages:list):
        # clear last used token
        self.last_used_token['completion_tokens'] = 0
        self.last_used_token['prompt_tokens'] = 0
        self.last_used_token['total_tokens'] = 0
        # chat
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            # max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        # print(response);exit()
        ans = response.choices[0].message.content
        ans = ans.replace('```json', '').replace('```', '').strip()
        self.last_used_token['completion_tokens'] = response.usage.completion_tokens
        # self.last_used_token['prompt_tokens'] = response.usage.prompt_tokens
        self.last_used_token['total_tokens'] = response.usage.total_tokens
        # count the used tokens
        self.used_token['completion_tokens'] += response.usage.completion_tokens
        # self.used_token['prompt_tokens'] += response.usage.prompt_tokens
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
                "url": f"{image_base64}", 
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