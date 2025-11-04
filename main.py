import yaml
from src.embeddings import Embedder
from src.faiss_index import FaissIndex
from src.cache_store import CacheStore
from src.retriever import Retriever
from src.assembler import assemble_evidence
from src.prompts import build_prompt
from src.llm_client import EchoLLM


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
    llm = EchoLLM()  # 后续切换到 ApiLLMClient

    return cfg, retriever, llm


if __name__ == "__main__":
    cfg, retriever, llm = bootstrap()

    # 1) 模拟注入一些历史
    retriever.add_history("user", "我叫李雷，以后叫我老李就行。", tags="偏好")
    retriever.add_history("assistant", "好的老李，我记住了。")
    retriever.add_history("user", "我把公司项目改到周三，别忘了。", tags="纠正")
    retriever.add_history("assistant", "收到，项目改到周三。")

    # 2) 新问题进来
    query = "这周项目是哪天？我叫什么来着？"
    items = retriever.search(query)
    evidence = assemble_evidence(
        items, cfg.get("retrieval", {}).get("max_context_chars")
    )
    prompt = build_prompt("耐心的助理", evidence, query)

    # 3) 生成（占位）
    answer = llm.generate(prompt, temperature=cfg.get("llm", {}).get("temperature"))
    print("\n===== PROMPT =====\n", prompt)
    print("\n===== ANSWER =====\n", answer)

    # 4) 保存索引（持久化）
    retriever.index.save()
