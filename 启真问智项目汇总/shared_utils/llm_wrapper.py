import os
from typing import Any, List, Optional
import base64

import dashscope
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

import logging

api_key = os.environ.get("DASHSCOPE_API_KEY")
if api_key:
    dashscope.api_key = api_key

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CustomChatDashScope(BaseChatModel):
    model: str = "qwen-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = 1200

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        prompt_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                prompt_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                prompt_messages.append({"role": "assistant", "content": msg.content})

        call_kwargs = dict(
            model=self.model,
            messages=prompt_messages,
            result_format="message",
            temperature=self.temperature,
            stream=False,
        )
        if self.max_tokens:
            call_kwargs["max_tokens"] = self.max_tokens
        call_kwargs.update(kwargs)
        response = dashscope.Generation.call(**call_kwargs)

        if hasattr(response, "status_code"):
            if response.status_code == 200:
                ai_content = response.output.choices[0]["message"]["content"]
                return AIMessage(content=ai_content)
        raise Exception(
                "DashScope API Error: Code {} , Message {}".format(
                    getattr(response, "code", "unknown"), getattr(response, "message", "unknown")
                )
            )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        ai_msg = self._call(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    @property
    def _llm_type(self) -> str:
        return "custom_chat_dashscope_wrapper"


class CustomVisionChatDashScope(BaseChatModel):
    model: str = "qwen-vl-max"
    temperature: float = 0.7
    max_tokens: Optional[int] = 1200

    def _encode_image_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        image_path: Optional[str] = None,
        **kwargs: Any,
    ) -> AIMessage:
        prompt_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                prompt_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                prompt_messages.append({"role": "assistant", "content": msg.content})

        try:
            mm_kwargs = dict(
                model=self.model,
                messages=prompt_messages,
                temperature=self.temperature,
                timeout=30,
            )
            if self.max_tokens:
                mm_kwargs["max_tokens"] = self.max_tokens
            mm_kwargs.update(kwargs)

            if image_path:
                import mimetypes
                import io
                from PIL import Image

                mime_type, _ = mimetypes.guess_type(image_path)
                if mime_type is None:
                    mime_type = "image/jpeg"
                with Image.open(image_path) as img:
                    max_dim = 1024
                    if max(img.size) > max_dim:
                        ratio = max_dim / float(max(img.size))
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.LANCZOS)
                    buffered = io.BytesIO()
                    save_format = mime_type.split("/")[-1].upper()
                    if save_format == "JPG":
                        save_format = "JPEG"
                    try:
                        img.save(buffered, format=save_format)
                    except Exception:
                        buffered = io.BytesIO()
                        img = img.convert('RGB')
                        img.save(buffered, format='JPEG')
                        mime_type = 'image/jpeg'
                    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                # Overwrite last user message with multimodal content
                for i in range(len(prompt_messages)-1, -1, -1):
                    if prompt_messages[i]["role"] == "user":
                        prompt_messages[i] = {"role": "user", "content": [
                            {"text": prompt_messages[i]["content"]},
                            {"image": f"data:{mime_type};base64,{encoded_image}"}
                        ]}
                        break

            response = dashscope.MultiModalConversation.call(**{
                "model": self.model,
                "messages": prompt_messages,
                "temperature": self.temperature,
                "timeout": 30,
                **({"max_tokens": self.max_tokens} if self.max_tokens else {}),
            })

            if hasattr(response, "status_code") and response.status_code == 200:
                ai_content = response.output.choices[0]["message"]["content"]
                if isinstance(ai_content, list):
                    try:
                        text_parts: List[str] = []
                        for part in ai_content:
                            if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
                                text_parts.append(part["text"])
                            else:
                                text_parts.append(str(part))
                        ai_content = " ".join([t for t in text_parts if t])
                    except Exception:
                        ai_content = str(ai_content)
                return AIMessage(content=ai_content)
            else:
                raise Exception("DashScope Vision API returned unexpected response format")
        except Exception as e:
            logging.error(f"视觉API调用失败: {e}")
            # Fallback to text-only API
            text_messages = []
            for msg_data in prompt_messages:
                if isinstance(msg_data.get("content"), list):
                    text_parts = [item.get("text", "") for item in msg_data["content"] if isinstance(item, dict) and "text" in item]
                    text_content = " ".join(text_parts)
                    text_messages.append({"role": msg_data["role"], "content": text_content})
                else:
                    text_messages.append(msg_data)

            response = dashscope.Generation.call(**{
                "model": "qwen-turbo",
                "messages": text_messages,
                "result_format": "message",
                "temperature": self.temperature,
                "stream": False,
                "timeout": 30,
                **({"max_tokens": self.max_tokens} if self.max_tokens else {}),
            })

            if hasattr(response, "status_code") and response.status_code == 200:
                ai_content = response.output.choices[0]["message"]["content"]
                logging.info("回退到文本模式成功")
                return AIMessage(content=f"[注意：图片分析功能暂时不可用，以下是基于文本的回复]\n\n{ai_content}")
            else:
                raise Exception("文本模式API调用也失败了")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        image_path: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatResult:
        ai_msg = self._call(messages, stop=stop, image_path=image_path, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def call_with_image(
        self,
        text: str,
        image_path: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=text))
        result = self._call(messages, image_path=image_path)
        return result.content

    @property
    def _llm_type(self) -> str:
        return "custom_vision_chat_dashscope_wrapper"


