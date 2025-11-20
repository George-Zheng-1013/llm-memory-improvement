from .history_store import HistoryStore
import yaml

CFG_PATH = r"E:\\task\\学校任务\\nlp项目\\llm-memory-improvement\\config.yaml"


def view_all_sessions():
    """查看所有会话"""
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    db_path = cfg.get("storage", {}).get("history_db_path")
    store = HistoryStore(db_path=db_path)

    sessions = store.get_all_sessions()

    if not sessions:
        print("还没有任何对话记录")
        return

    print("\n=== 所有会话列表 ===\n")
    for session_id, start_time, total_turns in sessions:
        print(f"会话ID: {session_id}")
        print(f"开始时间: {start_time}")
        print(f"对话轮数: {total_turns}")
        print("-" * 50)


def view_session_detail(session_id: str):
    """查看指定会话的详细对话记录"""
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    db_path = cfg.get("storage", {}).get("history_db_path")
    store = HistoryStore(db_path=db_path)

    history = store.get_session_history(session_id)

    if not history:
        print(f"未找到会话: {session_id}")
        return

    print(f"\n=== 会话详情: {session_id} ===\n")

    current_turn = None
    for turn_num, role, content, timestamp, created_at in history:
        if current_turn != turn_num:
            if current_turn is not None:
                print("-" * 50)
            current_turn = turn_num
            print(f"\n第 {turn_num} 轮对话 [{created_at}]:")

        role_name = "用户" if role == "user" else "助手"
        print(f"{role_name}: {content}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 查看指定会话
        session_id = sys.argv[1]
        view_session_detail(session_id)
    else:
        # 查看所有会话
        view_all_sessions()
        print(
            "\n提示: 使用 'python view_history.py <session_id>' 查看指定会话的详细记录"
        )
