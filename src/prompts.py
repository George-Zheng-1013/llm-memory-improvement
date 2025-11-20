# src/prompts.py
#这个版本实现了混合记忆，并稍微放宽了 Prompt 限制。
from typing import List, Tuple, Set
from .history_store import HistoryStore

def _format_history_content(history_list: List[Tuple], current_citation_index: int) -> Tuple[List[str], int]:
    """格式化历史记录"""
    formatted_parts = []
    # 按轮次排序
    history_sorted = sorted(history_list, key=lambda x: x[0]) 

    for turn_num, role, content, _, created_at in history_sorted:
        clean_content = " ".join(content.strip().split())
        if not clean_content: continue
        
        role_name = "用户" if role == "user" else "助手"
        formatted_parts.append(f"[证据#{current_citation_index}] [{created_at}] [{role_name}]: {clean_content}")
        current_citation_index += 1
        
    return formatted_parts, current_citation_index

def build_prompt(system_role: str, query: str, evidence: str = "") -> str:
    """
    构建 Prompt。
    修改策略：允许模型在证据不足时使用自身知识，但优先使用证据。
    """
    system = (
        f"系统指令：你是{system_role}。\n"
        f"1. 请优先参考提供的【历史对话记忆】来回答，保持上下文一致性。\n"
        f"2. 如果历史记忆中包含答案，**必须引用**[证据#编号]。\n"
        f"3. 如果历史记忆中没有相关信息，**请使用你自己的知识回答**，不要编造历史。\n"
    )
    task = f"用户问题：{query}"
    context = f"历史对话记忆:\n{evidence}" if evidence else "历史对话记忆：无"

    prompt = (
        f"{system}\n"
        f"{task}\n"
        f"{context}\n"
        f"请基于上述信息回答。"
    )
    return prompt

def get_evidence(
    history_store: HistoryStore,
    session_id: str,
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.5,
    recent_n: int = 2 # 新增：强制包含最近 N 轮
) -> str:
    """
    混合检索：短期记忆 (最近 N 轮) + 长期记忆 (语义检索)
    """
    evidence_parts = []
    collected_turn_numbers: Set[int] = set()
    current_citation_index = 1
    
    # --- 1. 获取短期记忆 (最近 N 轮，保证对话流畅性) ---
    # 我们先获取最近的记录 (不经过筛选)
    # 注意：这里的 limit 需要稍微大一点以确保覆盖 user+assistant
    recent_history_raw = history_store.get_session_history(session_id, limit=recent_n)
    
    for item in recent_history_raw:
        # item[0] 是 turn_number
        collected_turn_numbers.add(item[0])

    # --- 2. 获取长期记忆 (语义检索) ---
    # 只有当历史足够长时才检索，避免重复
    semantic_results = history_store.search_history_index(
        query, top_k=top_k, similarity_threshold=similarity_threshold
    )
    
    for turn_id, score in semantic_results:
        try:
            t_id_parts = turn_id.rsplit('_', 1)
            if len(t_id_parts) == 2:
                 s_id, t_num = t_id_parts
                 # 无论是否是当前 session，都可以作为长期记忆检索进来
                 # 这里我们为了简单，假设是同 session 或跨 session 都可以
                 collected_turn_numbers.add(int(t_num))
        except ValueError:
            continue
            
    if not collected_turn_numbers:
        return ""

    # --- 3. 统一获取内容并去重 ---
    # 获取当前 session 的历史 (如果需要跨 session，history_store 需要新增 get_turns_by_ids 方法)
    # 这里简化处理：只获取当前 session 的相关轮次
    all_history = history_store.get_session_history(session_id)
    
    filtered_history = [
        h for h in all_history if h[0] in collected_turn_numbers
    ]
    
    if not filtered_history:
        return ""

    # --- 4. 格式化 ---
    # 我们将它们区分为 "近期上下文" 和 "相关历史回忆" 也可以，但合并在一起通常模型也能理解
    # 这里直接合并
    evidence_parts.append("--- 对话记忆 (短期+语义检索) ---")
    formatted_parts, _ = _format_history_content(filtered_history, 1)
    evidence_parts.extend(formatted_parts)

    return "\n".join(evidence_parts)