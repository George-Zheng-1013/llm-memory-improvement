from src.llm_client import ApiLLMClient
from src.prompts import build_prompt, get_evidence_with_retrieval
from src.history_store import HistoryStore
from src.encoder import encode_and_save_turn
import yaml
from typing import Optional

CFG_PATH = r"config.yaml"


def print_sep(char: str = "─", title: Optional[str] = None, width: int = 60):
    """打印紧凑的一行分隔符，带可选标题（更少换行）。"""
    if title:
        # 紧凑标题样式：─── 标题 ───
        print(f"{char*3} {title} {char*3}")
    else:
        print(char * width)


def show_evidence(evidence: str):
    """结构化显示检索到的证据（紧凑模式）。"""
    if not evidence or not evidence.strip():
        print("[检索] 无相关历史证据")
        return
    print("【检索到的相关历史证据】")
    print(evidence.strip())
    print("【证据结束】")


def bootstrap():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("Config loaded.")
    llm = ApiLLMClient(
        api_key=cfg.get("llm", {}).get("api_key"),
        base_url=cfg.get("llm", {}).get("base_url"),
        model=cfg.get("llm", {}).get("model"),
    )

    # 初始化历史存储
    db_path = cfg.get("storage", {}).get(
        "history_db_path", "data/conversation_history.db"
    )
    history_store = HistoryStore(db_path=db_path)

    return cfg, llm, history_store


def chat_first_turn(cfg, llm, history_store):
    """首次对话流程"""
    print_sep("=", "开始对话")

    # 开始新的会话
    session_id = history_store.start_session()
    print(f"会话: {session_id[:8]}")
    turn_number = 1

    # 获取用户输入
    user_input = input("您: ").strip()

    if user_input.lower() in ["quit", "exit", "退出"]:
        print("对话已结束")
        return

    if not user_input:
        print("输入不能为空")
        return

    # 首轮对话（不检索）
    try:
        print_sep("-", "生成回答")
        answer = llm.generate(
            build_prompt("耐心的助理", user_input),
            temperature=cfg.get("llm", {}).get("temperature"),
        )
        print(f"助手: {answer}")
        print_sep("-")

        # 保存与编码
        db_path = cfg.get("storage", {}).get(
            "history_db_path", "data/conversation_history.db"
        )
        history_store.save_turn(session_id, turn_number, user_input, answer)
        encode_and_save_turn(db_path, session_id, turn_number)

    except Exception as e:
        print(f"✗ 错误: {e}")
        return

    chat_loop(cfg, llm, history_store, session_id, turn_number)


def chat_loop(cfg, llm, history_store, session_id, turn_number):
    db_path = cfg.get("storage", {}).get(
        "history_db_path", "data/conversation_history.db"
    )

    while True:
        # 获取用户输入
        user_input = input("您: ").strip()

        if user_input.lower() in ["quit", "exit", "退出"]:
            print(f"\n✓ 对话已结束，本次会话共 {turn_number} 轮")
            break

        if not user_input:
            continue

        # 增加轮数
        turn_number += 1

        # 使用语义检索获取最相关的证据
        print_sep()
        print(f"回合 {turn_number} — 检索中")
        evidence = get_evidence_with_retrieval(db_path, session_id, user_input, top_k=5)

        # 显示检索到的证据（若有）
        show_evidence(evidence)

        # 生成回答
        try:
            prompt = build_prompt("耐心的助理", user_input, evidence)
            answer = llm.generate(
                prompt, temperature=cfg.get("llm", {}).get("temperature")
            )
            print(f"助手: {answer}")
            print_sep()

            # 保存与编码
            history_store.save_turn(session_id, turn_number, user_input, answer)
            encode_and_save_turn(db_path, session_id, turn_number)

        except Exception as e:
            print(f"✗ 错误: {e}")
            turn_number -= 1  # 出错则不计入轮数


if __name__ == "__main__":
    cfg, llm, history_store = bootstrap()

    # 启动连续对话
    chat_first_turn(cfg, llm, history_store)
