from src.llm_client import ApiLLMClient
from src.prompts import build_prompt
import yaml

CFG_PATH = "config.yaml"


def bootstrap():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("Loaded config:", cfg)
    llm = ApiLLMClient(
        api_key=cfg.get("llm", {}).get("api_key"),
        base_url=cfg.get("llm", {}).get("base_url"),
        model=cfg.get("llm", {}).get("model"),
    )
    return cfg, llm


def chat_loop(cfg, llm):
    """连续对话循环，支持历史检索"""
    print("\n=== 开始对话 (输入 'quit' 或 'exit' 退出) ===\n")

    while True:
        # 获取用户输入
        user_input = input("用户: ").strip()

        if user_input.lower() in ["quit", "exit", "退出"]:
            print("对话结束")
            break

        if not user_input:
            continue

        # 3) 构建 prompt
        prompt = build_prompt("耐心的助理", user_input)
        print(
            f"----------<构建的Prompt>----------\n{prompt}\n----------<构建的Prompt>----------"
        )

        # 4) 调用大模型生成回复
        try:
            answer = llm.generate(
                prompt, temperature=cfg.get("llm", {}).get("temperature")
            )
            print(f"助手: {answer}")

        except Exception as e:
            print(f"错误: {str(e)}")
            continue
        print("----------<本轮对话结束>----------\n")


if __name__ == "__main__":
    cfg, llm = bootstrap()

    # 启动连续对话
    chat_loop(cfg, llm)
