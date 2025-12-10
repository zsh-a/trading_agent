
import os
import base64
import mimetypes
from typing import Any, List, Optional, Dict
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)
from llama_index.llms.openai import OpenAI
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from llama_index.core.llms.callbacks import llm_chat_callback

class ModelScopeLLM(OpenAI):
    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-30B-A3B-Thinking",
        api_key: Optional[str] = None,
        api_base: str = "https://api-inference.modelscope.cn/v1",
        **kwargs: Any,
    ) -> None:
        # 使用 MODELSCOPE_API_KEY 或 OPENAI_API_KEY
        api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        super().__init__(model=model, api_key=api_key, api_base=api_base, **kwargs)
        self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)

    def _convert_messages_to_openai(self, messages: List[ChatMessage]) -> List[Dict]:
        openai_msgs = []
        for msg in messages:
            role = msg.role.value
            if isinstance(msg.content, list):
                # Handle multimodal
                content_list = []
                for block in msg.content:
                    if block.__class__.__name__ == 'TextBlock':
                         content_list.append({"type": "text", "text": block.text})
                    elif block.__class__.__name__ == 'ImageBlock':
                        # Convert local path to base64 data URL
                        path = block.path # type: ignore
                        if os.path.exists(path):
                            mime_type, _ = mimetypes.guess_type(path)
                            if not mime_type:
                                mime_type = "image/png"
                            with open(path, "rb") as f:
                                base64_image = base64.b64encode(f.read()).decode('utf-8')
                            url = f"data:{mime_type};base64,{base64_image}"
                            content_list.append({
                                "type": "image_url",
                                "image_url": {"url": url}
                            })
                        else:
                             # Fallback if path doesn't exist (e.g. url)
                             content_list.append({
                                "type": "image_url",
                                "image_url": {"url": path}
                            })
                openai_msgs.append({"role": role, "content": content_list})
            else:
                openai_msgs.append({"role": role, "content": msg.content})
        return openai_msgs

    @llm_chat_callback()
    async def achat(
        self,
        messages: List[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        # 装饰器 @llm_chat_callback() 会自动捕获 messages 参数并传递给 tracing 系统
        # messages 参数会被传递给 callback_manager.on_event_start() 和 dispatcher.event()
        # 确保 callback_manager 已正确初始化（由装饰器处理）
        openai_messages = self._convert_messages_to_openai(messages)
        
        model_kwargs = self._get_model_kwargs(**kwargs)
        if "model" in model_kwargs:
            del model_kwargs["model"]
        
        # 调用 API
        response_stream = await self._async_client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            stream=True,
            extra_body={"enable_thinking": True},
            **model_kwargs,
        )

        full_content = ""
        full_reasoning = ""
        done_reasoning = False
        print("\n=== Model Thinking ===")
        
        async for chunk in response_stream:
            # Type annotation helper
            chunk: ChatCompletionChunk = chunk # type: ignore
            if not chunk.choices:
                continue
                
            delta = chunk.choices[0].delta
            
            # Access reasoning_content safely (custom field for this model)
            # The python SDK might not expose it in the typed object if it's extra field,
            # so we access via .model_extra or dict if needed, or getattr
            reasoning_chunk = getattr(delta, 'reasoning_content', '') or ''
            answer_chunk = delta.content or ''
            
            if reasoning_chunk:
                print(reasoning_chunk, end='', flush=True)
                full_reasoning += reasoning_chunk
            elif answer_chunk:
                if not done_reasoning:
                    print('\n=== Final Answer ===\n')
                    done_reasoning = True
                print(answer_chunk, end='', flush=True)
                full_content += answer_chunk
                
        print("\n")
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, 
                content=full_content,
                additional_kwargs={"reasoning": full_reasoning}
            ),
            raw=None, # Optional
        )
