import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from typing import List, Tuple
import torch

# 初始化模型
bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
bi_encoder.max_seq_length = 256

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def blob_to_np(blob: bytes) -> np.ndarray:
    """将二进制blob转换回numpy数组"""
    import io

    bio = io.BytesIO(blob)
    return np.load(bio)

#原版
# def retrieve_top_k_relevant(
#     db_path: str, session_id: str, query: str, top_k: int = 4
# ) -> List[Tuple[str, str, float]]:
#     """
#     检索与查询最相关的前k条对话记录，使用bi-encoder和cross-encoder重排序

#     Args:
#         db_path: SQLite数据库文件路径
#         session_id: 会话ID
#         query: 查询文本
#         top_k: 返回的前k条记录

#     Returns:
#         [(user_content, assistant_content, score), ...] 排序后的对话记录
#     """
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     try:
#         # 获取该会话的所有对话
#         cursor.execute(
#             """
#             SELECT h.id, h.turn_number, h.role, h.content
#             FROM conversation_history h
#             WHERE h.session_id = ?
#             ORDER BY h.turn_number ASC
#             """,
#             (session_id,),
#         )
#         all_conversations = cursor.fetchall()

#         if not all_conversations:
#             return []

#         # 将对话按turn分组
#         turns_dict = {}
#         for row_id, turn_num, role, content in all_conversations:
#             if turn_num not in turns_dict:
#                 turns_dict[turn_num] = {}
#             turns_dict[turn_num][role] = content

#         # 构建用于检索的文本列表
#         turn_numbers = sorted(turns_dict.keys())
#         passages = []
#         turn_num_list = []

#         for turn_num in turn_numbers:
#             turn_data = turns_dict[turn_num]
#             if "user" in turn_data and "assistant" in turn_data:
#                 combined_text = f"{turn_data['user']} {turn_data['assistant']}"
#                 passages.append(combined_text)
#                 turn_num_list.append(turn_num)

#         if not passages:
#             return []

#         # ========== Bi-Encoder 语义搜索 ==========
#         query_embedding = bi_encoder.encode(query, convert_to_tensor=True)

#         # 对所有passages进行编码
#         passages_embeddings = bi_encoder.encode(passages, convert_to_tensor=True)

#         # 计算相似度
#         cos_scores = util.pytorch_cos_sim(query_embedding, passages_embeddings)[0]
#         top_results = torch.topk(cos_scores, k=min(top_k * 2, len(passages)))

#         # ========== Cross-Encoder 重排序 ==========
#         bi_encoder_hits = []
#         for score, idx in zip(top_results[0], top_results[1]):
#             bi_encoder_hits.append(
#                 {
#                     "turn_num": turn_num_list[idx],
#                     "passage": passages[idx],
#                     "bi_score": score.item(),
#                 }
#             )

#         # 使用cross-encoder重排序
#         cross_inp = [[query, hit["passage"]] for hit in bi_encoder_hits]
#         cross_scores = cross_encoder.predict(cross_inp)

#         # 添加cross-encoder得分
#         for idx, hit in enumerate(bi_encoder_hits):
#             hit["cross_score"] = cross_scores[idx]

#         # 按cross-encoder得分排序
#         bi_encoder_hits = sorted(
#             bi_encoder_hits, key=lambda x: x["cross_score"], reverse=True
#         )

#         # 获取前k条结果
#         top_hits = bi_encoder_hits[:top_k]

#         # 构建返回结果
#         results = []
#         for hit in top_hits:
#             turn_num = hit["turn_num"]
#             user_content = turns_dict[turn_num].get("user", "")
#             assistant_content = turns_dict[turn_num].get("assistant", "")
#             score = hit["cross_score"]
#             results.append((user_content, assistant_content, score))

#         return results

#     except Exception as e:
#         print(f"检索失败: {str(e)}")
#         return []
#     finally:
#         conn.close()

#第一轮权重增强版
def retrieve_top_k_relevant(
    db_path: str, session_id: str, query: str, top_k: int = 4
) -> List[Tuple[str, str, float]]:
    """
    检索与查询最相关的前k条对话记录，使用bi-encoder和cross-encoder重排序
    *** 增强：对第一轮 (T1 轮) 进行角色加权 (Role Boosting) ***
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 角色加权因子：将 T1 轮次的得分提高 50%
    ROLE_BOOST_FACTOR = 1.5 

    try:
        # 1. 获取该会话的所有对话和 embedding (略... 与原代码一致)
        # ... [原代码：获取 turns_dict, all_embeddings, all_ids, all_turn_nums] ...
        cursor.execute(
            """
            SELECT 
                h.id, h.turn_number, h.role, h.content, 
                e.embedding, h.timestamp
            FROM conversation_history h
            JOIN conversation_embeddings e ON h.id = e.convo_row_id
            WHERE h.session_id = ?
            ORDER BY h.timestamp ASC
            """,
            (session_id,),
        )
        
        raw_hits = cursor.fetchall()
        
        # 整理数据结构
        turns_dict = {}
        embeddings = []
        embedding_map = {} # 存储 {embedding_id: (id, turn_num, role, content)}

        for id, turn_num, role, content, emb_blob, timestamp in raw_hits:
            if turn_num not in turns_dict:
                turns_dict[turn_num] = {"timestamp": timestamp}
            turns_dict[turn_num][role] = content
            
            emb_np = blob_to_np(emb_blob)
            embeddings.append(emb_np)
            embedding_map[len(embeddings) - 1] = {
                "id": id, 
                "turn_num": turn_num, 
                "role": role, 
                "content": content,
                "user_content": turns_dict[turn_num].get("user", ""), # 方便后续获取 user/assistant content
                "assistant_content": turns_dict[turn_num].get("assistant", ""),
            }


        if not embeddings:
            return []

        # 2. Bi-Encoder 检索 (余弦相似度)
        query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
        
        # *** 修复重点：获取 query_embedding 所在的设备 ***
        # 确保 embeddings_tensor 和 query_embedding 位于同一设备
        target_device = query_embedding.device

        # 将所有 embedding 的 NumPy 数组转换为 Tensor，并移动到 target_device
        embeddings_tensor = torch.stack([
            torch.from_numpy(emb).to(target_device) # <--- 关键修改：.to(target_device)
            for emb in embeddings
        ])
        
        # 计算相似度
        # 现在 cos_sim 应该能成功运行，因为两个 Tensor 都在 target_device 上
        cos_scores = util.cos_sim(query_embedding, embeddings_tensor)[0]
        
        # 获取 Top-K Bi-Encoder 结果
        bi_encoder_hits = []
        # 确保至少能检索到 top_k*2 的数量，以便重排序
        top_indices = torch.topk(cos_scores, min(len(cos_scores), top_k * 2), largest=True) 

        for score, idx in zip(top_indices.values.tolist(), top_indices.indices.tolist()):
            map_data = embedding_map[idx]
            # 这里记录的是 bi-encoder 原始得分
            bi_encoder_hits.append({
                "score": score, 
                "turn_num": map_data["turn_num"],
                "user_content": map_data["user_content"],
                "assistant_content": map_data["assistant_content"],
                "raw_content": map_data["content"]
            })

        # 3. Cross-Encoder 重排序
        # Cross-Encoder 需要 [query, context] 对
        cross_inp = [
            [query, f"用户: {hit['user_content']} 助手: {hit['assistant_content']}"]
            for hit in bi_encoder_hits
        ]
        
        cross_scores = cross_encoder.predict(cross_inp)

        # 4. 添加 Cross-Encoder 得分并进行角色加权
        for idx, hit in enumerate(bi_encoder_hits):
            cross_score = cross_scores[idx]
            
            # *** 关键增强：对 T1 轮次的得分进行加权 ***
            if hit["turn_num"] == 1:
                # 确保 T1 轮次能被优先召回
                hit["cross_score"] = cross_score * ROLE_BOOST_FACTOR 
            else:
                hit["cross_score"] = cross_score


        # 5. 按 Cross-Encoder 得分排序
        bi_encoder_hits = sorted(
            bi_encoder_hits, key=lambda x: x["cross_score"], reverse=True
        )

        # 6. 获取前k条结果并格式化
        top_hits = bi_encoder_hits[:top_k]

        results = []
        for hit in top_hits:
            results.append((hit["user_content"], hit["assistant_content"], hit["cross_score"]))

        return results

    except Exception as e:
        print(f"检索失败: {str(e)}")
        # 打印详细错误信息，帮助调试
        import traceback
        traceback.print_exc()
        return []
    finally:
        conn.close()

def format_retrieved_results(results: List[Tuple[str, str, float]]) -> str:
    """
    格式化检索结果为可用于prompt的字符串

    Args:
        results: [(user_content, assistant_content, score), ...]

    Returns:
        格式化的字符串
    """
    if not results:
        return ""

    formatted = []
    for idx, (user_content, assistant_content, score) in enumerate(results, 1):
        formatted.append(f"[证据#{idx}] (相关度: {score:.3f})")
        formatted.append(f"用户: {user_content}")
        formatted.append(f"助手: {assistant_content}")
        formatted.append("")

    return "\n".join(formatted)
