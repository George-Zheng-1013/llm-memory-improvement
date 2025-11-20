import requests
import yaml

CFG_PATH = r"E:\\task\\学校任务\\nlp项目\\llm-memory-improvement\\config.yaml"


class LLMClient:
    def generate(self, prompt: str, temperature: float = 0.2, **kwargs) -> str:
        raise NotImplementedError


class ApiLLMClient(LLMClient):

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
    ):
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.api_key = config.get("api_key", api_key)
        self.base_url = config.get("base_url", base_url)
        self.model = config.get("model", model)

    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": kwargs.get("stream", False),
                "max_tokens": kwargs.get("max_tokens", 4096),
                "enable_thinking": kwargs.get("enable_thinking", False),
                "thinking_budget": kwargs.get("thinking_budget", 4096),
                "min_p": kwargs.get("min_p", 0.05),
                "stop": kwargs.get("stop", None),
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.7),
                "top_k": kwargs.get("top_k", 50),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
                "n": kwargs.get("n", 1),
                "response_format": kwargs.get("response_format", {"type": "text"}),
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}")
