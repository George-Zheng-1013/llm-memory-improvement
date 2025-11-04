import yaml
from src.embeddings import Embedder
from src.faiss_index import FaissIndex
from src.cache_store import CacheStore
from src.retriever import Retriever
from src.assembler import assemble_evidence
from src.prompts import build_prompt
from src.llm_client import EchoLLM, ApiLLMClient

CFG_PATH = "config.yaml"

def bootstrap():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("Loaded config:", cfg)
    embedder = Embedder(
        cfg.get("embedding", {}).get("model"),
        device=cfg.get("embedding", {}).get("device", "auto"),
        batch_size=cfg.get("embedding", {}).get("batch_size", 64),
    )
    index = FaissIndex(
        cfg.get("faiss", {}).get("dim"), cfg.get("faiss", {}).get("index_path")
    )
    store = CacheStore(cfg.get("storage", {}).get("sqlite_path"))
    retriever = Retriever(embedder, index, store, topk=cfg.get("faiss", {}).get("topk"))
    llm = ApiLLMClient(
        api_key=cfg.get("llm", {}).get("api_key"),
        base_url=cfg.get("llm", {}).get("base_url"),
        model=cfg.get("llm", {}).get("model"),
    )

    return cfg, retriever, llm


def chat_loop(cfg, retriever, llm):
    """连续对话循环，支持历史检索"""
    print("\n=== 开始对话 (输入 'quit' 或 'exit' 退出) ===\n")

    while True:
        # 获取用户输入
        user_input = input("用户: ").strip()

        if user_input.lower() in ["quit", "exit", "退出"]:
            print("对话结束，保存索引...")
            retriever.index.save()
            break

        if not user_input:
            continue

        # 1) 保存用户输入到历史
        retriever.add_history("user", user_input)

        # 2) 检索相关历史
        items = retriever.search(user_input)
        evidence = assemble_evidence(
            items, cfg.get("retrieval", {}).get("max_context_chars", 1200)
        )

        # 3) 构建 prompt
        prompt = build_prompt("耐心的助理", evidence, user_input)

        # 4) 调用大模型生成回复
        try:
            answer = llm.generate(
                prompt, temperature=cfg.get("llm", {}).get("temperature")
            )
            print(f"\n助手: {answer}\n")

            # 5) 保存助手回复到历史
            retriever.add_history("assistant", answer)

        except Exception as e:
            print(f"\n错误: {str(e)}\n")
            continue


if __name__ == "__main__":
    cfg, retriever, llm = bootstrap()

    # 可选：初始化一些示例历史（仅首次运行或测试时使用）
    retriever.add_history("user", "我叫李雷，以后叫我老李就行。", tags="偏好")
    retriever.add_history("assistant", "好的老李，我记住了。")

    # 启动连续对话
    chat_loop(cfg, retriever, llm)
