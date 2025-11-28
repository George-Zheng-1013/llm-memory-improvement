from typing import List
from .history_store import HistoryStore
from .retriever import retrieve_top_k_relevant, format_retrieved_results
from typing import Optional
#原版
# def build_prompt(system_role: str, query: str, evidence: str = "") -> str:
#     system = (
#         f"系统指令：你是{system_role}。如果历史与当前指令冲突，以'用户纠正'为准；"
#         f"当证据不足时要明确说明，不要编造。回答时尽量引用[证据#编号]。"
#     )
#     context = f"历史证据:\n{evidence}" if evidence else "历史证据：未收集到历史证据"
#     task = f"用户问题：{query}"

#     prompt = (
#         f"{system}\n{task}\n{context}\n请基于上述证据回答，并在关键处标注[证据#x]。"
#     )
#     return prompt

#基于新提示词的新版
from typing import Optional
from .history_store import HistoryStore
from .retriever import retrieve_top_k_relevant, format_retrieved_results

def build_prompt(system_role: str, query: str, evidence: Optional[str] = None) -> str:
    """
    构建 LLM 最终接收的 Prompt 消息。通用化以支持 B 模块（事实）和 C 模块（角色）。
    """
    
    evidence = evidence if evidence else ""
    has_retrieved_evidence = bool(evidence.strip())
    
    # --- 1. 定义核心系统指令（通用化且更强硬） ---
    system_directive = (
        f"系统指令：你是{system_role}。如果历史与当前指令冲突，以'用户纠正'为准；"
        f"当证据不足时要明确说明，不要编造。"
        f"**你正在接受记忆力测试。用户最终的问题（T7轮）将针对你在T1轮次接收的**角色设定或关键事实**。**"
    )

    # --- 2. 定义状态声明和引用规则 ---
    if has_retrieved_evidence:
        # RAG 模式：召回成功。命令模型必须使用召回的信息。
        state_declaration = "**【证据状态】: 已召回 T1 关键信息（角色/事实）。**"
        citation_rule = (
            "请基于召回的历史证据中的 T1 关键信息（角色/事实）来回答用户的问题，以保持连贯性和一致性。"
            "**对于所有与 T1 设定的角色或事实相关的 T7 问题，你必须表现出记忆。**"
            "如果你直接使用了 T1 证据中的具体文字，请在关键处标注[证据#x]；否则，无需标注。"
        )
    else:
        # Baseline 模式 / RAG 召回失败：强制拒绝回答 T1 记忆目标。
        state_declaration = "**【证据状态】: 未召回任何检索证据。**"
        citation_rule = (
            "请基于上述指令和证据回答。**由于你无法从历史证据中找到 T1 轮次的关键信息（角色设定或关键事实），**"
            "**因此你必须拒绝回答用户所有依赖于该 T1 信息的后续问题。**"
            "**对于 T1 相关的问题，请严格回答：'由于历史证据中缺乏相关的角色或事实设定，我无法回答该问题。'**"
            "对于与 T1 设定完全无关的通用知识问题，你可以使用你的通用知识回答，但无需引用[证据#x]。"
        )

    # --- 3. 构建最终 Prompt ---
    context = f"历史证据:\n{evidence}" if has_retrieved_evidence else "历史证据：无检索证据"
    task = f"用户问题：{query}"

    prompt = (
        f"{system_directive}\n\n"
        f"{task}\n\n"
        f"{context}\n\n"
        f"{state_declaration}\n" 
        f"{citation_rule}"
    )
    return prompt

def get_evidence(history_store: HistoryStore, session_id: str, n_turns: int = 5) -> str:
    recent_history = history_store.get_recent_history(session_id, n_turns=n_turns)
    return recent_history.strip() if isinstance(recent_history, str) else recent_history


def get_evidence_with_retrieval(
    db_path: str, session_id: str, query: str, top_k: int = 5
) -> str:
    """
    使用语义检索获取最相关的证据

    Args:
        db_path: SQLite数据库文件路径
        session_id: 会话ID
        query: 用户查询
        top_k: 返回的前k条记录

    Returns:
        格式化的证据字符串
    """
    results = retrieve_top_k_relevant(db_path, session_id, query, top_k=top_k)
    formatted = format_retrieved_results(results)
    return formatted.strip()
