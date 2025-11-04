from typing import Optional
from openai import OpenAI


class LLMClient:
    def generate(self, prompt: str, temperature: float = 0.2, **kwargs) -> str:
        raise NotImplementedError


class EchoLLM(LLMClient):
    def generate(self, prompt: str, temperature: float = 0.2, **kwargs) -> str:
        # 占位：截断回显，保证流程可跑
        return prompt[-800:]


class ApiLLMClient(LLMClient):

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "qwen/qwen-2.5-7b-instruct",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(self, prompt: str, temperature: float = 0.2, **kwargs) -> str:
        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": kwargs.get("site_url", ""),
                    "X-Title": kwargs.get("site_name", ""),
                },
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["site_url", "site_name"]
                },
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}")
