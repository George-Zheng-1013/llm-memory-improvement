# ==================== 依赖导入 ====================
import yaml
import time
import random
import os
import json
import pandas as pd
from typing import Optional, Tuple, List, Dict
import re
# 假设这些模块文件存在于您的项目中
from src.llm_client import ApiLLMClient
from src.prompts import build_prompt, get_evidence_with_retrieval
from src.history_store import HistoryStore
from src.encoder import encode_and_save_turn

try:
    from datasets import load_dataset
    print("正在加载 CMRC2018 数据集用于自动测试...")
    cmrc2018_dataset = load_dataset("cmrc2018", split="train") 
    print(f"✓ 数据集加载完成，共 {len(cmrc2018_dataset)} 条数据。")
except Exception as e:
    print(f"✗ 警告: 无法加载 CMRC2018 数据集: {e}")
    cmrc2018_dataset = None


CFG_PATH = r"config.yaml"
# 全局配置
AUTO_NOISE_ROUNDS = 5 
TOP_K_VALUES = [1,3,5,6,7,8,9,10]
EVAL_TEMPERATURE = 0.0 # 评估器的温度

# 全局结果收集列表
GLOBAL_RESULTS: List[Dict] = []

# 复合得分权重 (用户要求)
WEIGHT_B_ACC = 0.5 
WEIGHT_C_RC = 0.5
WEIGHT_TOTAL_CHARS = 0.001 # 假设一个惩罚系数


# ==================== 通用辅助函数 ====================

def print_sep(char: str = "─", title: Optional[str] = None, width: int = 60):
    """打印紧凑的一行分隔符，带可选标题。"""
    if title:
        print(f"{char*3} {title} {char*3}")
    else:
        print(char * width)


def show_evidence(evidence: str):
    """结构化显示检索到的证据（紧凑模式，限制长度）。"""
    if not evidence or not evidence.strip():
        print("[检索] 无相关历史证据")
        return
    print("【检索到的相关历史证据】")
    print(evidence.strip()[:200] + ("..." if len(evidence.strip()) > 200 else ""))
    print("【证据结束】")


def bootstrap():
    """初始化配置、LLM客户端（生成和评估）和历史存储。"""
    if not os.path.exists(CFG_PATH):
        raise FileNotFoundError(f"配置文件未找到: {CFG_PATH}")
        
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("Config loaded.")
    
    # 1. 生成 LLM 客户端 (用于对话生成)
    llm = ApiLLMClient(
        api_key=cfg.get("llm", {}).get("api_key", "YOUR_API_KEY"),
        base_url=cfg.get("llm", {}).get("base_url", "YOUR_BASE_URL"),
        model=cfg.get("llm", {}).get("model1", "YOUR_MODEL_NAME"),
    )
    
    # 2. 评估 LLM 客户端 (用于自动判分，低温度保证客观性)
    eval_llm = ApiLLMClient(
        api_key=cfg.get("llm", {}).get("api_key", "YOUR_API_KEY"),
        base_url=cfg.get("llm", {}).get("base_url", "YOUR_BASE_URL"),
        model=cfg.get("llm", {}).get("model1", "YOUR_MODEL_NAME"),
    )

    db_path = cfg.get("storage", {}).get(
        "history_db_path", "data/conversation_history.db"
    )
    history_store = HistoryStore(db_path=db_path)

    return cfg, llm, eval_llm, history_store


def sample_user_query_and_answer() -> Optional[str]:
    """用于噪声注入，返回带有文章的完整查询字符串。"""
    global cmrc2018_dataset
    if not cmrc2018_dataset:
        return None
        
    idx = random.randint(0, len(cmrc2018_dataset) - 1)
    sample = cmrc2018_dataset[idx]
    question = sample.get("question", "")
    
    user_query = question
    
    return user_query


# ==================== 核心修改：增加 top_k 参数和返回值 ====================

def run_turn(cfg, llm, history_store, session_id, turn_number: int, user_input: str, 
             is_noise: bool = False, use_retrieval: bool = True, is_rag_mode: bool = True, top_k: int = 5) -> Tuple[int, str, int]:
    """执行单个对话回合，包含检索、生成和存储，并返回答案和字符数。"""
    db_path = cfg.get("storage", {}).get(
        "history_db_path", "data/conversation_history.db"
    )
    
    mode_text = "噪声注入" if is_noise else ("检索增强 (RAG)" if is_rag_mode and use_retrieval else ("无检索 (Baseline)" if not is_rag_mode else "首轮无检索"))
    if is_rag_mode and use_retrieval:
        mode_text += f" [K={top_k}]"
        
    print_sep("-", f"回合 {turn_number} — {mode_text}")
    print(f"用户: {user_input[:80]}...") 
    
    evidence = ""
    if is_rag_mode and use_retrieval:
        evidence = get_evidence_with_retrieval(db_path, session_id, user_input, top_k=top_k)
    show_evidence(evidence)

    try:
        prompt = build_prompt("耐心的助理", user_input, evidence)
        answer = llm.generate(
            prompt, temperature=cfg.get("llm", {}).get("temperature")
        )
        print(f"助手: {answer}")
        print_sep()

        history_store.save_turn(session_id, turn_number, user_input, answer)
        encode_and_save_turn(db_path, session_id, turn_number)
        
        return turn_number + 1, answer, len(answer) # <--- 返回答案和字符数
    
    except Exception as e:
        print(f"✗ 错误: LLM调用或存储失败: {e}")
        return turn_number, "", 0


# ==================== T/F评估函数 ====================

# def evaluate_answer(eval_llm: ApiLLMClient, module: str, question: str, answer: str, gold_standard: str) -> float:
#     """使用 LLM 评估器对答案进行评分 (ACC 或 RC)。"""
    
#     if module == "B":
#         metric_name = "准确率 (ACC)"
#         # 黄金标准是 T1 设定的事实
#         prompt = (
#             f"你是一个严格的评分员。请判断以下对话回答是否准确地召回了会话最初设定的关键事实。请只输出一个布尔值：True/False。\n\n"
#             f"【关键事实/黄金标准】: {gold_standard}\n"
#             f"【最终提问】: {question}\n"
#             f"【模型回答】: {answer}\n"
#             f"评分结果 (True/False): "
#         )
#     elif module == "C":
#         metric_name = "角色一致性 (RC)"
#         # 黄金标准是 T1 设定的角色/上下文
#         prompt = (
#             f"你是一个严格的评分员。请判断以下回答是否完全符合会话最初设定的角色/上下文。请只输出一个布尔值：True/False。\n\n"
#             f"【设定角色/上下文】: {gold_standard}\n"
#             f"【最终提问】: {question}\n"
#             f"【模型回答】: {answer}\n"
#             f"评分结果 (True/False): "
#         )
#     else: # 模块 A
#         metric_name = "准确率 (ACC)"
#         # 模块 A T20的评估标准是召回关键事实
#         prompt = (
#             f"你是一个严格的评分员。请判断以下回答是否准确地召回了会话中设定的关键信息。请只输出一个布尔值：True/False。\n\n"
#             f"【设定的关键信息】: {gold_standard}\n"
#             f"【最终提问】: {question}\n"
#             f"【模型回答】: {answer}\n"
#             f"评分结果 (True/False): "
#         )

#     try:
#         raw_response = eval_llm.generate(prompt, temperature=EVAL_TEMPERATURE)
#         clean_response = raw_response.strip().lower()
        
#         if "true" in clean_response or "是" in clean_response or "1" in clean_response:
#             score = 1.0
#         elif "false" in clean_response or "否" in clean_response or "0" in clean_response:
#             score = 0.0
#         else:
#             print(f" [WARN] 评估器解析失败，结果: {clean_response[:30]}...")
#             score = 0.0
            
#         print(f" [评估] {metric_name} 得分: {score}")
#         return score
    
#     except Exception as e:
#         print(f" [ERROR] LLM 评估失败: {e}")
#         return 0.0

#[0,1]区间打分函数，精确到小数点后两位
def evaluate_answer(eval_llm: ApiLLMClient, module: str, question: str, answer: str, gold_standard: str) -> float:
    """使用 LLM 评估器对答案进行 0-1 区间（精确到两位小数）的评分 (ACC 或 RC)。"""
    
    if module == "B":
        metric_name = "准确率 (ACC)"
        # 黄金标准是 T1 设定的事实
        prompt = (
            f"你是一个严格的评分员。请根据以下关键事实/黄金标准，判断模型回答的准确性。"
            f"请在 0.00（完全不准确）到 1.00（完全准确）的区间内给出一个分数，精确到两位小数。"
            f"请只输出最终分数。\n\n"
            f"【关键事实/黄金标准】: {gold_standard}\n"
            f"【最终提问】: {question}\n"
            f"【模型回答】: {answer}\n"
            f"评分结果 (0.00-1.00): "
        )
    elif module == "C":
        metric_name = "角色一致性 (RC)"
        # 黄金标准是 T1 设定的角色/上下文
        prompt = (
            f"你是一个严格的评分员。请根据设定的角色/上下文，判断模型回答的角色一致性。"
            f"请在 0.00（完全不符合）到 1.00（完全符合）的区间内给出一个分数，精确到两位小数。"
            f"请只输出最终分数。\n\n"
            f"【设定角色/上下文】: {gold_standard}\n"
            f"【最终提问】: {question}\n"
            f"【模型回答】: {answer}\n"
            f"评分结果 (0.00-1.00): "
        )
    else: # 模块 A
        metric_name = "准确率 (ACC)"
        # 模块 A T20的评估标准是召回关键事实
        prompt = (
            f"你是一个严格的评分员。请根据设定的关键信息，判断模型回答的准确性。"
            f"请在 0.00（完全不准确）到 1.00（完全准确）的区间内给出一个分数，精确到两位小数。"
            f"请只输出最终分数。\n\n"
            f"【设定的关键信息】: {gold_standard}\n"
            f"【最终提问】: {question}\n"
            f"【模型回答】: {answer}\n"
            f"评分结果 (0.00-1.00): "
        )

    try:
        # 假设 EVAL_TEMPERATURE 变量在函数外部已定义
        raw_response = eval_llm.generate(prompt, temperature=EVAL_TEMPERATURE)
        clean_response = raw_response.strip()

        # 使用正则表达式查找第一个浮点数（包括整数、带小数点的数字）
        # 匹配模式：可选的负号，至少一个数字，可选的小数点和后面的数字
        match = re.search(r'(\d*\.?\d+)', clean_response)
        
        score = 0.0
        if match:
            try:
                # 尝试将匹配到的字符串转换为浮点数
                extracted_score = float(match.group(1))
                
                # 确保分数在 [0.00, 1.00] 范围内
                if extracted_score > 1.0:
                    score = 1.0
                elif extracted_score < 0.0:
                    score = 0.0
                else:
                    score = extracted_score
                    
            except ValueError:
                # 如果转换失败（非常规情况），则保留 0.0
                print(f" [WARN] 评估器解析失败，无法将 '{match.group(1)}' 转换为数字。")
        else:
            # 如果没有找到任何数字
            print(f" [WARN] 评估器解析失败，未找到任何数字。原始结果: {clean_response[:30]}...")
            
        # 将最终得分精确到两位小数
        final_score = round(score, 2)
        
        print(f" [评估] {metric_name} 原始输出: {clean_response[:30]}..., 得分: {final_score}")
        return final_score
    
    except Exception as e:
        print(f" [ERROR] LLM 评估失败: {e}")
        return 0.0

# ==================== 结果分析函数 ====================

def calculate_module_results(module_name: str, results: List[Dict]):
    """计算并打印单个模块的结果分析报告。"""
    
    module_data = pd.DataFrame([r for r in results if r['module'] == module_name])
    
    if module_data.empty:
        print(f"✗ 模块 {module_name} 无有效结果数据。")
        return

    metric = 'ACC_score' if module_name in ['A', 'B'] else 'RC_score'
    metric_label = '准确率 (ACC)' if module_name in ['A', 'B'] else '角色一致性 (RC)'

    print_sep("=", f"--- 模块 {module_name} 结果分析 ---", 70)

    if module_name == 'A':
        # 模块 A: Top-K 对比
        topk_results = module_data.groupby('top_k')[metric].mean().sort_index()
        print(f"模块 A (Top-K 对比) - {metric_label} 结果:")
        print(topk_results.to_string(float_format='%.4f'))
        print("\n结论: Top-K 值对召回准确率的影响如上表所示。")
        
    elif module_name in ['B', 'C']:
        # 模块 B/C: RAG vs Baseline 对比
        comparison = module_data.groupby('mode')[metric].mean()
        
        base_score = comparison.get('baseline', 0.0)
        rag_score = comparison.get('rag', 0.0)
        
        if base_score > 0:
            improvement = (rag_score - base_score) / base_score * 100
        else:
            improvement = float('inf') if rag_score > 0 else 0.0
        
        print(f"【{module_name} 模块：{metric_label} 对比】")
        print(f"  Baseline ({metric_label}): {base_score:.4f}")
        print(f"  RAG ({metric_label})     : {rag_score:.4f}")
        print(f"  相对提升                : {improvement:.2f}%")
        
    print_sep("=", f"--- 模块 {module_name} 结果分析结束 ---", 70)


# test_auto_try.py 文件中，替换此函数:

def calculate_final_composite_score(results: List[Dict]):
    """计算最终的复合总分，并同时展示 RAG 和 Baseline 的结果。"""
    
    final_data = pd.DataFrame(results)
    
    # ------------------ 1. 筛选 RAG 数据并计算 ------------------
    rag_data = final_data[
        (final_data['mode'] == 'rag') & 
        (final_data['module'].isin(['B', 'C']))
    ].copy()
    
    if rag_data.empty:
        print_sep("=", "--- 最终复合总分计算结果 ---", 70)
        print("✗ RAG 模式下 B/C 模块数据不足，无法计算 RAG 总分。")
        rag_b_acc = rag_c_rc = total_chars_mean = final_score = 0.0
    else:
        rag_b_acc = rag_data[rag_data['module'] == 'B']['ACC_score'].mean() if not rag_data[rag_data['module'] == 'B'].empty else 0.0
        rag_c_rc = rag_data[rag_data['module'] == 'C']['RC_score'].mean() if not rag_data[rag_data['module'] == 'C'].empty else 0.0
        total_chars_mean = rag_data['total_chars'].mean() if not rag_data.empty else 0.0
        
        # RAG Final Score = (w_B * ACC_RAG) + (w_C * RC_RAG) - (w_len * total_chars_mean)
        final_score = (
            WEIGHT_B_ACC * rag_b_acc + 
            WEIGHT_C_RC * rag_c_rc - 
            WEIGHT_TOTAL_CHARS * total_chars_mean
        )

    # ------------------ 2. 筛选 Baseline 数据并计算 ------------------
    baseline_data = final_data[
        (final_data['mode'] == 'baseline') & 
        (final_data['module'].isin(['B', 'C']))
    ].copy()
    
    if baseline_data.empty:
        print_sep("=", "--- 最终复合总分计算结果 ---", 70)
        print("✗ Baseline 模式下 B/C 模块数据不足，无法计算 Baseline 总分。")
        baseline_b_acc = baseline_c_rc = baseline_total_chars_mean = baseline_score = 0.0
    else:
        baseline_b_acc = baseline_data[baseline_data['module'] == 'B']['ACC_score'].mean() if not baseline_data[baseline_data['module'] == 'B'].empty else 0.0
        baseline_c_rc = baseline_data[baseline_data['module'] == 'C']['RC_score'].mean() if not baseline_data[baseline_data['module'] == 'C'].empty else 0.0
        baseline_total_chars_mean = baseline_data['total_chars'].mean() if not baseline_data.empty else 0.0

        # Baseline Final Score = (w_B * ACC_Base) + (w_C * RC_Base) - (w_len * total_chars_mean_Base)
        baseline_score = (
            WEIGHT_B_ACC * baseline_b_acc + 
            WEIGHT_C_RC * baseline_c_rc - 
            WEIGHT_TOTAL_CHARS * baseline_total_chars_mean
        )
    
    # ------------------ 3. 打印最终报告 ------------------
    print_sep("=", "--- 最终复合总分计算结果 ---", 70)
    print("【权重配置】:")
    print(f"  B 模块 ACC 权重 (ω_B): {WEIGHT_B_ACC}")
    print(f"  C 模块 RC 权重 (ω_C): {WEIGHT_C_RC}")
    print(f"  字符数惩罚权重 (ω_len): {WEIGHT_TOTAL_CHARS}")
    
    print("\n【RAG 模式 (增强方案) 平均指标】:")
    print(f"  B 模块平均准确率 (ACC): {rag_b_acc:.4f}")
    print(f"  C 模块平均一致性 (RC): {rag_c_rc:.4f}")
    print(f"  平均答案字符数 (TotalChars): {total_chars_mean:.2f}")
    print(f"  RAG Final Score = ({WEIGHT_B_ACC}*{rag_b_acc:.4f}) + ({WEIGHT_C_RC}*{rag_c_rc:.4f}) - ({WEIGHT_TOTAL_CHARS}*{total_chars_mean:.2f})")
    print(f"  RAG FINAL SCORE: {final_score:.4f}")
    
    print("\n【Baseline 模式 (原生 LLM) 平均指标】:")
    print(f"  B 模块平均准确率 (ACC): {baseline_b_acc:.4f}")
    print(f"  C 模块平均一致性 (RC): {baseline_c_rc:.4f}")
    print(f"  平均答案字符数 (TotalChars): {baseline_total_chars_mean:.2f}")
    print(f"  Baseline Final Score = ({WEIGHT_B_ACC}*{baseline_b_acc:.4f}) + ({WEIGHT_C_RC}*{baseline_c_rc:.4f}) - ({WEIGHT_TOTAL_CHARS}*{baseline_total_chars_mean:.2f})")
    print(f"  BASELINE FINAL SCORE: {baseline_score:.4f}")

    if final_score > baseline_score:
        print(f"\n★ 结论: RAG 模式总分高于 Baseline，性能提升约 {(final_score - baseline_score):.4f}。")
    elif final_score < baseline_score:
        print(f"\n☆ 结论: Baseline 模式总分高于 RAG，需要优化 RAG 检索/重排序以减少负面影响。")
    else:
        print("\n结论: RAG 与 Baseline 性能持平。")

    print_sep()


# ==================== 模块 A：RAG Top-K 准确率测试 (已修改) ====================

def run_module_a(cfg, llm, eval_llm, history_store):
    """
    执行 A 模块的完整流程：循环测试不同的 Top-K 值。
    """
    A_TOTAL_TURNS = 20
    A_MANUAL_INPUT_TURNS = [1, 5, 9, 13, 17]
    for current_k in TOP_K_VALUES: 
        
        print_sep("=", f"--- 开始执行 A 模块：Top-K 准确率测试 (K={current_k}, 20 轮) ---", 70)
        session_id = history_store.start_session()
        current_turn = 1
        t1_input = "" # 存储 T1 输入作为黄金标准
        
        print(f"A 模块会话: {session_id[:8]} (K={current_k}, 目标: {A_TOTAL_TURNS} 轮)")

        while current_turn <= A_TOTAL_TURNS:
            
            is_manual_set_turn = current_turn in A_MANUAL_INPUT_TURNS
            is_final_test_turn = current_turn == A_TOTAL_TURNS
            
            answer = ""
            char_count = 0
            user_input = ""

            if is_manual_set_turn:
                print_sep("=", f"A 模块 - T{current_turn}: 手动输入关键信息")
                print("请输入一个关键事实或角色设定：")
                user_input = input(f"您 (T{current_turn}): ").strip()
                
                if not user_input:
                    print("手动输入不能为空，测试提前结束。")
                    break
                
                if current_turn == 1:
                    t1_input = user_input # 记录 T1 输入
                    
                use_retrieval = (current_turn != 1) 
                new_turn_number, answer, char_count = run_turn(
                    cfg, llm, history_store, session_id, current_turn, user_input, 
                    use_retrieval=use_retrieval, is_rag_mode=True, top_k=current_k
                )

            elif is_final_test_turn:
                print_sep("=", f"A 模块 - T{current_turn}: 最终测试问题")
                print(f"已完成 {current_turn - 1} 轮对话。请提问与 T1/T5/T9/T13/T17 轮关键信息相关的问题。")
                
                user_input = input(f"您 (T{current_turn}): ").strip()
                if not user_input:
                    print("最终测试问题不能为空，测试结束。")
                    break

                new_turn_number, answer, char_count = run_turn(
                    cfg, llm, history_store, session_id, current_turn, user_input, 
                    use_retrieval=True, is_rag_mode=True, top_k=current_k
                )
                
                # --- 结果分析：T20 轮次评分与数据收集 ---
                if new_turn_number > current_turn:
                    acc_score = evaluate_answer(
                        eval_llm, 
                        "A", 
                        user_input, # T20 question
                        answer, 
                        t1_input # 使用 T1 输入作为召回的关键事实
                    )
                    GLOBAL_RESULTS.append({
                        'module': 'A',
                        'mode': 'rag',
                        'session_id': session_id,
                        'top_k': current_k,
                        'ACC_score': acc_score,
                        'RC_score': float('nan'),
                        'total_chars': char_count,
                        'gold_standard': t1_input
                    })
                
            else:
                user_input = sample_user_query_and_answer()

                if user_input is None:
                    print("数据集抽取失败，停止 A 模块自动测试。")
                    break
                    
                print_sep("=", f"A 模块 - T{current_turn}: 自动噪声注入")
                new_turn_number, answer, char_count = run_turn(
                    cfg, llm, history_store, session_id, current_turn, user_input, 
                    is_noise=True, use_retrieval=True, is_rag_mode=True, top_k=current_k
                )
            
            if new_turn_number > current_turn:
                current_turn = new_turn_number
                if current_turn <= A_TOTAL_TURNS:
                    time.sleep(1) 
            else:
                print(f"第 {current_turn} 轮运行失败，当前 Top-K 测试提前结束。")
                break

        print(f"\n✓ A 模块 Top-K={current_k} 测试结束，本次会话共 {current_turn - 1} 轮")
        print_sep("=", f"--- A 模块 Top-K={current_k} 执行完成 ---", 70)
        time.sleep(2)
    
    # 模块结束后打印分析
    calculate_module_results('A', GLOBAL_RESULTS)


# ==================== 模块 B：长对话召回率测试 (RAG vs Baseline) ====================

def chat_first_turn_B(cfg, llm, history_store, session_id, is_rag_mode: bool) -> Tuple[Optional[str], int, Optional[str]]:
    """B 模块 T1: 手动输入初始关键信息（不检索）。"""
    mode_label = "RAG" if is_rag_mode else "BASELINE"
    print_sep("=", f"B 模块：长对话召回率测试 ({mode_label} - T1)")
    print(f"B 模块会话: {session_id[:8]}")
    print("第一轮 (T1)：请手动输入一个关键事实/信息用于召回测试（例如：你最喜欢的颜色是蓝色）。")

    user_input = input("您 (T1 - 关键信息): ").strip()

    if not user_input:
        print("输入不能为空，退出 B 模块。")
        return None, 0, None

    turn_number = 1
    new_turn_number, _, _ = run_turn(cfg, llm, history_store, session_id, turn_number, user_input, use_retrieval=False, is_rag_mode=is_rag_mode)
    
    if new_turn_number > turn_number:
        return session_id, new_turn_number, user_input
    return None, 0, None


def chat_loop_B(cfg, llm, eval_llm, history_store, session_id, turn_number, t1_input: str, is_rag_mode: bool):
    """B 模块 T2+：自动噪声注入 -> 最终手动测试（检索）。"""
    
    current_turn = turn_number
    mode_label = "RAG" if is_rag_mode else "BASELINE"
    
    # PHASE 1: 自动噪声循环
    target_auto_turn = current_turn + AUTO_NOISE_ROUNDS 
    # ... (噪声注入逻辑不变) ...
    while current_turn < target_auto_turn:
        user_input = sample_user_query_and_answer()
        if user_input is None: break
        
        current_turn, _, _ = run_turn(cfg, llm, history_store, session_id, current_turn, user_input, is_noise=True, is_rag_mode=is_rag_mode)
        if current_turn <= target_auto_turn: time.sleep(1)
        else: break
            
    # PHASE 2: 最终手动输入轮次
    print_sep("=", f"B 模块 - 最终测试回合：请手动输入问题 ({mode_label})")
    print(f"已完成 {current_turn - 1} 轮对话。请提问与 T1 轮的关键信息 ('{t1_input[:20]}...') 相关的问题。")

    while True:
        user_input_final = input(f"您 (第 {current_turn} 轮 - {mode_label}): ").strip()

        if user_input_final.lower() in ["quit", "exit", "退出"]:
            print(f"\n✓ B 模块 ({mode_label}) 测试结束，本次会话共 {current_turn - 1} 轮")
            return current_turn - 1 

        if not user_input_final:
            continue
            
        new_turn_number, answer, char_count = run_turn(cfg, llm, history_store, session_id, current_turn, user_input_final, is_rag_mode=is_rag_mode)

        if new_turn_number > current_turn:
            # --- 结果分析：T_Final 轮次评分与数据收集 ---
            acc_score = evaluate_answer(
                eval_llm, 
                "B", 
                user_input_final, 
                answer, 
                t1_input # T1 input is the gold standard
            )
            
            GLOBAL_RESULTS.append({
                'module': 'B',
                'mode': 'rag' if is_rag_mode else 'baseline',
                'session_id': session_id,
                'top_k': 6 if is_rag_mode else float('nan'), 
                'ACC_score': acc_score,
                'RC_score': float('nan'), 
                'total_chars': char_count,
                'gold_standard': t1_input
            })
            
            print(f"\n✓ B 模块 ({mode_label}) 测试结束，本次会话共 {new_turn_number - 1} 轮")
            return new_turn_number - 1
        
        current_turn = new_turn_number 
        
    return current_turn - 1


def run_module_b(cfg, llm, eval_llm, history_store):
    """执行 B 模块的完整流程: RAG vs Baseline 对比测试"""
    
    test_modes = {"rag": True, "baseline": False}
    
    for mode_name, is_rag_mode in test_modes.items():
        print_sep("=", f"--- 开始执行 B 模块：长对话召回率测试 ({mode_name.upper()} 模式) ---", 70)
        session_id = history_store.start_session()
        
        result = chat_first_turn_B(cfg, llm, history_store, session_id, is_rag_mode) 
        
        if result and result[1] > 0:
            session_id, next_turn, t1_input = result
            chat_loop_B(cfg, llm, eval_llm, history_store, session_id, next_turn, t1_input, is_rag_mode)
        
        print_sep("=", f"--- B 模块 ({mode_name.upper()} 模式) 执行完成 ---", 70)
        time.sleep(2)
    
    # 模块结束后打印分析
    calculate_module_results('B', GLOBAL_RESULTS)


# ==================== 模块 C：上下文角色一致性测试 (RAG vs Baseline) ====================

def chat_first_turn_C(cfg, llm, history_store, session_id, is_rag_mode: bool) -> Tuple[Optional[str], int, Optional[str]]:
    """C 模块 T1：手动输入关键事实/角色（不检索）。"""
    mode_label = "RAG" if is_rag_mode else "BASELINE"
    print_sep("=", f"C 模块：上下文角色一致性测试 ({mode_label} - T1)")
    print(f"C 模块会话: {session_id[:8]}")
    print("第一轮 (T1)：请手动输入一个关键角色或事实（例如：你是外科医生）")

    user_input = input("您 (T1): ").strip()
    if not user_input:
        print("输入不能为空，退出 C 模块。")
        return None, 0, None

    turn_number = 1
    new_turn_number, _, _ = run_turn(cfg, llm, history_store, session_id, turn_number, user_input, use_retrieval=False, is_rag_mode=is_rag_mode)
    
    if new_turn_number == turn_number:
        return None, 0, None
    
    return session_id, new_turn_number, user_input


def chat_loop_C(cfg, llm, eval_llm, history_store, session_id, turn_number, t1_input: str, is_rag_mode: bool):
    """C 模块 T2+：自动噪声注入 + 最终手动测试（检索）。"""
    
    target_auto_turn = turn_number + AUTO_NOISE_ROUNDS 
    mode_label = "RAG" if is_rag_mode else "BASELINE"

    # ... (噪声注入逻辑不变) ...
    while turn_number < target_auto_turn:
        user_input = sample_user_query_and_answer()
        if user_input is None: break
            
        turn_number, _, _ = run_turn(cfg, llm, history_store, session_id, turn_number, user_input, is_noise=True, is_rag_mode=is_rag_mode)
        if turn_number <= target_auto_turn: time.sleep(1)
        else: break
            
    print_sep("=", f"C 模块 - 最终测试回合：请手动输入问题 ({mode_label})")
    print(f"已完成 {turn_number - 1} 轮对话。请提问与 T1 轮相关的、需要模型保持角色的问题。")

    current_turn = turn_number
    while True:
        user_input = input(f"您 (第 {current_turn} 轮 - {mode_label}): ").strip()

        if user_input.lower() in ["quit", "exit", "退出"]:
            print(f"\n✓ C 模块 ({mode_label}) 测试结束，本次会话共 {current_turn - 1} 轮")
            return current_turn - 1 

        if not user_input:
            continue
            
        new_turn_number, answer, char_count = run_turn(cfg, llm, history_store, session_id, current_turn, user_input, is_rag_mode=is_rag_mode)

        if new_turn_number > current_turn:
            # --- 结果分析：T_Final 轮次评分与数据收集 ---
            rc_score = evaluate_answer(
                eval_llm, 
                "C", 
                user_input, 
                answer, 
                t1_input # T1 input is the gold standard (role setting)
            )
            
            GLOBAL_RESULTS.append({
                'module': 'C',
                'mode': 'rag' if is_rag_mode else 'baseline',
                'session_id': session_id,
                'top_k': 6 if is_rag_mode else float('nan'), 
                'ACC_score': float('nan'), 
                'RC_score': rc_score,
                'total_chars': char_count,
                'gold_standard': t1_input
            })
            
            print(f"\n✓ C 模块 ({mode_label}) 测试结束，本次会话共 {new_turn_number - 1} 轮")
            return new_turn_number - 1
        
        current_turn = new_turn_number 
        
    return current_turn - 1


def run_module_c(cfg, llm, eval_llm, history_store):
    """执行 C 模块的完整流程: RAG vs Baseline 对比测试"""
    
    test_modes = {"rag": True, "baseline": False}

    for mode_name, is_rag_mode in test_modes.items():
        print_sep("=", f"--- 开始执行 C 模块：上下文角色一致性测试 ({mode_name.upper()} 模式) ---", 70)
        session_id = history_store.start_session()
        
        result = chat_first_turn_C(cfg, llm, history_store, session_id, is_rag_mode)
        
        if result and result[1] > 0:
            session_id, next_turn, t1_input = result
            chat_loop_C(cfg, llm, eval_llm, history_store, session_id, next_turn, t1_input, is_rag_mode)
        
        print_sep("=", f"--- C 模块 ({mode_name.upper()} 模式) 执行完成 ---", 70)
        time.sleep(2)
        
    # 模块结束后打印分析
    calculate_module_results('C', GLOBAL_RESULTS)


# ==================== 主入口函数 (外层执行 ABC) ====================

if __name__ == "__main__":
    try:
        cfg, llm, eval_llm, history_store = bootstrap()
    except Exception as e:
        print(f"初始化失败: {e}")
        exit(1)
        
    print_sep("=", "--- RAG vs Baseline 增强记忆对比测试开始 ---", 80)
    
    # 1. 执行 A 模块 (RAG Top-K 对比)
    # run_module_a(cfg, llm, eval_llm, history_store)
    
    # 2. 执行 B 模块 (RAG vs Baseline 对比)
    run_module_b(cfg, llm, eval_llm, history_store)
    
    # 3. 执行 C 模块 (RAG vs Baseline 对比)
    run_module_c(cfg, llm, eval_llm, history_store)
    
    # 4. 计算最终复合总分
    calculate_final_composite_score(GLOBAL_RESULTS)
    
    print_sep("=", "--- 所有测试模块运行完毕 ---", 80)