import openai
import requests
import google.generativeai as genai
from openai import OpenAI
import os
from typing import List, Optional, Union, Dict, Any

class Claude:
    def __init__(self, api_key: str):
        """
        Claude 类的初始化方法
        :param api_key: Claude API 密钥
        """
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }

    def generate_reply(self, prompt: str, model: str = "claude-3-opus-20240229", max_tokens: int = 1024) -> str:
        """
        使用 Claude 模型生成回复
        :param prompt: 输入的提示语
        :param model: 使用的模型名称
        :param max_tokens: 最大生成的令牌数
        :return: 模型的生成回复
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()['content'][0]['text']
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_reply_with_files(
        self, 
        prompt: str, 
        files: List[str], 
        model: str = "claude-3-opus-20240229", 
        max_tokens: int = 1024
    ) -> str:
        """
        生成包含文件上传的回复
        :param prompt: 用户输入的提示
        :param files: 文件路径列表
        :param model: 使用的模型名称
        :param max_tokens: 最大生成的令牌数
        :return: 生成的回复内容
        """
        try:
            # 构建包含文件的消息内容
            message_content = []
            
            # 添加文本部分
            message_content.append({
                "type": "text",
                "text": prompt
            })
            
            # 添加文件部分
            for file_path in files:
                if not os.path.exists(file_path):
                    return f"Error: File not found: {file_path}"
                
                # 获取MIME类型
                file_type = self._get_mime_type(file_path)
                
                # 读取文件内容并进行base64编码
                with open(file_path, "rb") as f:
                    import base64
                    file_data = base64.b64encode(f.read()).decode('utf-8')
                
                message_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": file_type,
                        "data": file_data
                    }
                })
            
            # 构建API请求
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": message_content}],
                "max_tokens": max_tokens
            }
            
            response = requests.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()['content'][0]['text']
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _get_mime_type(self, file_path: str) -> str:
        """
        获取文件的MIME类型
        :param file_path: 文件路径
        :return: MIME类型
        """
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            # 默认为二进制流
            return "application/octet-stream"
        return mime_type



class ChatGPT:
    def __init__(self, api_key: str):
        """
        ChatGPT 类的初始化方法
        :param api_key: OpenAI API 密钥
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")
    
    def generate_reply(self, prompt: str, model: str = "gpt-4o", max_tokens: int = 150) -> str:
        """
        生成回复
        :param prompt: 用户输入的提示
        :param model: 使用的模型名称
        :param max_tokens: 最大生成的令牌数
        :return: 生成的回复内容
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_reply_with_files(
        self, 
        prompt: str, 
        files: List[str], 
        model: str = "gpt-4o", 
        max_tokens: int = 150
    ) -> str:
        """
        生成包含文件上传的回复
        :param prompt: 用户输入的提示
        :param files: 文件路径列表
        :param model: 使用的模型名称
        :param max_tokens: 最大生成的令牌数
        :return: 生成的回复内容
        """
        try:
            # 上传文件并获取文件ID
            file_ids = []
            for file_path in files:
                if not os.path.exists(file_path):
                    return f"Error: File not found: {file_path}"
                
                file = self.client.files.create(
                    file=open(file_path, "rb"),
                    purpose="assistants"
                )
                print(type(file))
                file_ids.append(file.id)
            
            # 创建包含文件的消息
            messages = [{"role": "user", "content": prompt}]
            
            # 创建带有文件的聊天完成
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                file_ids=file_ids
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    def upload_file(self, file_path: str) -> str:
        """
        上传单个文件
        :param file_path: 文件路径
        :return: 文件ID或错误信息
        """
        try:
            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"
            
            file = self.client.files.create(
                file=open(file_path, "rb"),
                purpose="assistants"
            )
            return file.id
        except Exception as e:
            return f"Error: {str(e)}"


class DeepSeek:
    def __init__(self, api_key: str):
        """
        DeepSeek 类的初始化方法
        :param api_key: DeepSeek API 密钥
        """
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def generate_reply(self, prompt: str, model: str = "deepseek-chat", max_tokens: int = 1024) -> str:
        """
        使用 DeepSeek 模型生成回复
        :param prompt: 输入的提示语
        :param model: 使用的模型名称
        :param max_tokens: 最大生成的令牌数
        :return: 模型的生成回复
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_reply_with_files(
        self, 
        prompt: str, 
        files: List[str], 
        model: str = "deepseek-chat", 
        max_tokens: int = 1024
    ) -> str:
        """
        生成包含文件上传的回复
        :param prompt: 用户输入的提示
        :param files: 文件路径列表
        :param model: 使用的模型名称
        :param max_tokens: 最大生成的令牌数
        :return: 生成的回复内容
        """
        try:
            # 构建包含文件的消息内容
            message_content = []
            
            # 添加文本部分
            message_content.append({
                "type": "text",
                "text": prompt
            })
            
            # 添加文件部分
            for file_path in files:
                if not os.path.exists(file_path):
                    return f"Error: File not found: {file_path}"
                
                # 获取MIME类型
                file_type = self._get_mime_type(file_path)
                
                # 读取文件内容并进行base64编码
                with open(file_path, "rb") as f:
                    import base64
                    file_data = base64.b64encode(f.read()).decode('utf-8')
                
                message_content.append({
                    "type": "image",
                    "image_url": {
                        "url": f"data:{file_type};base64,{file_data}"
                    }
                })
            
            # 构建API请求
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": message_content}],
                "max_tokens": max_tokens
            }
            
            response = requests.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _get_mime_type(self, file_path: str) -> str:
        """
        获取文件的MIME类型
        :param file_path: 文件路径
        :return: MIME类型
        """
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            # 默认为二进制流
            return "application/octet-stream"
        return mime_type


class Gemini:
    def __init__(self, api_key: str):
        """
        Gemini 类的初始化方法
        :param api_key: Gemini API 密钥
        """
        self.api_key = api_key
        # 初始化Google Generative AI
        genai.configure(api_key=self.api_key)
        
    def generate_reply(self, prompt: str, model: str = "gemini-pro", max_tokens: int = 1024) -> str:
        """
        使用 Gemini 模型生成回复
        :param prompt: 输入的提示语
        :param model: 使用的模型名称
        :param max_tokens: 最大生成的令牌数
        :return: 模型的生成回复
        """
        try:
            # 创建模型实例
            model_instance = genai.GenerativeModel(model)
            
            # 生成回复
            response = model_instance.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens}
            )
            
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_reply_with_files(
        self, 
        prompt: str, 
        files: List[str], 
        model: str = "gemini-pro-vision", 
        max_tokens: int = 1024
    ) -> str:
        """
        生成包含文件上传的回复
        :param prompt: 用户输入的提示
        :param files: 文件路径列表
        :param model: 使用的模型名称
        :param max_tokens: 最大生成的令牌数
        :return: 生成的回复内容
        """
        try:
            # 创建模型实例 (使用支持视觉的模型)
            model_instance = genai.GenerativeModel(model)
            
            # 准备内容列表
            contents = [prompt]
            
            # 添加图像文件
            for file_path in files:
                if not os.path.exists(file_path):
                    return f"Error: File not found: {file_path}"
                
                # 读取图像文件
                from PIL import Image
                image = Image.open(file_path)
                contents.append(image)
            
            # 生成回复
            response = model_instance.generate_content(
                contents,
                generation_config={"max_output_tokens": max_tokens}
            )
            
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"