# main.py

from src.llm_client import ApiLLMClient
from src.prompts import build_prompt, get_evidence
from src.history_store import HistoryStore
from src.embedding_utils import EMBEDDING_CLIENT 
import yaml
import os

CFG_PATH = r"config.yaml" 

def bootstrap():
    if not os.path.exists(CFG_PATH):
        raise FileNotFoundError(f"Config file not found at {CFG_PATH}.")
        
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("Loaded config:", cfg)
    
    llm = ApiLLMClient(
        api_key=cfg.get("llm", {}).get("api_key"),
        base_url=cfg.get("llm", {}).get("base_url"),
        model=cfg.get("llm", {}).get("model"),
    )

    db_path = cfg.get("storage", {}).get("history_db_path", "./cache/conversation_history.db")
    faiss_dir = cfg.get("storage", {}).get("faiss_index_dir", "./cache")

    try:
        history_store = HistoryStore(db_path=db_path, faiss_index_dir=faiss_dir)
        # 核心修复：调用重建索引，确保之前的记录被加载
        history_store.rebuild_faiss_index()
    except Exception as e:
        print(f"Error initializing HistoryStore: {e}")
        history_store = None

    # 不再初始化 External Retriever
        
    return cfg, llm, history_store

def chat_first_turn(cfg, llm, history_store):
    print("\n=== 开始对话 (输入 'quit' 退出) ===\n")
    session_id = history_store.start_session()
    print(f"会话ID: {session_id}")
    
    chat_loop(cfg, llm, history_store, session_id, 0)

def chat_loop(cfg, llm, history_store, session_id, last_turn_number):
    turn_number = last_turn_number
    rag_cfg = cfg.get("rag", {})
    system_role = cfg.get("llm", {}).get("system_role", "助理")
    
    while True:
        user_input = input("用户: ").strip()
        if user_input.lower() in ["quit", "exit", "退出"]:
            history_store.update_session_total_turns(session_id, turn_number)
            print("对话结束。")
            break
        if not user_input: continue

        turn_number += 1

        # 对话式 RAG：从历史中检索证据
        evidence = ""
        try:
            evidence = get_evidence(
                history_store, 
                session_id, 
                user_input, 
                top_k=rag_cfg.get("top_k", 5), 
                similarity_threshold=rag_cfg.get("similarity_threshold", 0.5)
            )
        except Exception as e:
            print(f"Warn: History retrieval failed: {e}")
            
        prompt = build_prompt(system_role, user_input, evidence)
        print(f"--- Context ---\n{evidence}\n----------------")

        try:
            answer = llm.generate(prompt, temperature=cfg.get("llm", {}).get("temperature", 0.7))
            print(f"助手: {answer}")
            history_store.save_turn(session_id, turn_number, user_input, answer)
        except Exception as e:
            print(f"Error: {e}")
            turn_number -= 1
        print("--------------------------\n")

if __name__ == "__main__":
    cfg, llm, history_store = bootstrap()
    if history_store:
        chat_first_turn(cfg, llm, history_store)