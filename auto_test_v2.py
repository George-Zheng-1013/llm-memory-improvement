# auto_test_v3.py
import sqlite3
import os
import json
import time
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import random

from src.llm_client import ApiLLMClient
from src.history_store import HistoryStore
from src.encoder import encode_and_save_turn
from src.retriever import retrieve_top_k_relevant, format_retrieved_results
from src.prompts import build_prompt
from datasets import load_dataset

# 使用 CMRC2018 train split（轻量）
cmrc2018_dataset = load_dataset("cmrc2018", split="train")

# ============= 全局配置 =============
TEST_DB_PATH = "data/test_history.db"
CONFIG_PATH = "config.yaml"

NUM_RUNS = 5
ROLE_BOOST = 0.25
GEN_TEMPERATURE = 0.1
EVAL_TEMPERATURE = 0.0

EMBEDDING_MODEL = "text-embedding-3-large"
os.environ["EMBEDDING_MODEL"] = EMBEDDING_MODEL

# ============= 工具函数 =============
def extract_text(resp):
    if isinstance(resp, str):
        return resp
    if hasattr(resp, "text"):
        return resp.text
    if hasattr(resp, "output_text"):
        return resp.output_text
    if hasattr(resp, "content"):
        return resp.content
    if isinstance(resp, dict):
        if "text" in resp:
            return resp["text"]
        if "content" in resp:
            return resp["content"]
    return str(resp)


def llm_score_0_2(client: ApiLLMClient, prompt: str) -> float:
    strict_prompt = (
        "请严格遵守输出格式要求：只输出一个数字（0、1 或 2），"
        "不允许任何多余解释或文本，换行也算多余。仅输出数字字符并结束。\n\n"
        + prompt
    )
    try:
        raw = client.generate(strict_prompt, temperature=EVAL_TEMPERATURE, max_tokens=4, stop=["\n"])
        text = extract_text(raw).strip()
        match = re.search(r"\b([0-2])\b", text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(2.0, score))
        else:
            match2 = re.search(r"([0-2](?:\.\d+)?)", text)
            if match2:
                return float(match2.group(1))
            return 1.0
    except Exception as e:
        print(f"[ERR] llm_score_0_2 调用失败: {e}")
        return 1.0


def calculate_char_length(evidence_text: str) -> int:
    return len(evidence_text)


def evaluate_with_llm(scenario: Dict, query: str, answer: str, client: ApiLLMClient) -> Dict[str, float]:
    expected_keywords = scenario.get("expected_keywords", [])
    role_keywords = scenario.get("role_keywords", [])
    scenario_id = scenario["id"]

    accuracy_prompt = f"""
你是严格的上下文记忆评估员。请根据用户问题、期望关键词和模型回答，给回答打分（0-2）。
场景ID: {scenario_id}
用户查询: {query}
期望关键字: {expected_keywords}
模型回答: {answer}
评分（0-2）：
"""
    accuracy_score = llm_score_0_2(client, accuracy_prompt)

    role_score = 2.0
    if scenario_id == "role_consistency":
        role_prompt = f"""
你是严格的角色一致性评估员。请根据期望角色关键词判断模型回答是否遵守角色（0-2）。
期望关键词: {role_keywords}
模型回答: {answer}
评分（0-2）：
"""
        role_score = llm_score_0_2(client, role_prompt)

    return {
        "accuracy_llm": accuracy_score / 2.0,
        "role_consistency_llm": role_score / 2.0
    }


# ============= 初始化系统 =============
def init_system_from_config():
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass
    store = HistoryStore(db_path=TEST_DB_PATH)

    try:
        import yaml
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        llm_cfg = cfg.get("llm", {})
        client = ApiLLMClient(
            api_key=llm_cfg.get("api_key"),
            base_url=llm_cfg.get("base_url"),
            model=llm_cfg.get("model1")
        )
    except Exception:
        client = ApiLLMClient(api_key=None, base_url=None, model=None)

    try:
        setattr(client, "embedding_model", EMBEDDING_MODEL)
    except Exception:
        pass
    os.environ["EMBEDDING_MODEL"] = EMBEDDING_MODEL

    return store, client


# ============= 角色加权函数 =============
def apply_role_boost(retrieved: List[Tuple], role_keywords: List[str], boost: float) -> List[Tuple]:
    boosted = []
    for r in retrieved:
        content = str(r[1]) if len(r) > 1 else ""
        score = float(r[2]) if len(r) > 2 else 0.0
        hit = any(kw.lower() in content.lower() for kw in role_keywords)
        new_score = min(score + boost if hit else score, 1.0)
        if len(r) >= 3:
            new_r = (r[0], r[1], new_score)
        else:
            new_r = (r[0], r[1] if len(r) > 1 else "", new_score)
        boosted.append(new_r)
    return sorted(boosted, key=lambda x: x[2], reverse=True)


# ============= 数据抽样函数 =============
def sample_user_query_and_answer():
    idx = random.randint(0, len(cmrc2018_dataset) - 1)
    sample = cmrc2018_dataset[idx]
    context = sample.get("context", "")
    question = sample.get("question", "")
    answers = sample.get("answers", {}).get("text", []) if sample.get("answers") else []
    answer = answers[0] if answers else "参考答案不可用"
    user_query = f"请阅读下面的内容，并回答问题：\n\n{context}\n\n问题：{question}"
    return user_query, answer


# ============= 场景定义 =============
long_history = []
for i in range(10):
    q, a = sample_user_query_and_answer()
    long_history.append({"user": q, "assistant": a})

noise_history = []
for i in range(40):
    q, a = sample_user_query_and_answer()
    noise_history.append({"user": f"噪声 {i+1}：" + q, "assistant": a})


def build_memory_dialog(memory_fact: str, noise_turns: int = 80) -> List[Dict[str, str]]:
    dialog = []
    noise_templates = [
        "今天天气不错，你怎么看？",
        "你觉得人工智能会取代程序员吗？",
        "我今天吃了咖喱，味道不错。",
        "你看过那部新电影吗？",
        "我昨天在公园散步，看到很多人。",
        "你喜欢猫还是狗？",
        "你平时喜欢听哪种音乐？",
        "今天有点困，可能要喝杯咖啡。"
    ]
    for i in range(noise_turns):
        u = random.choice(noise_templates)
        a = "嗯，明白了。"
        dialog.append({"user": f"噪声 {i+1}：" + u, "assistant": a})
    dialog.append({"user": f"重要信息：{memory_fact}", "assistant": "好的，我会记住这条重要信息。"})
    return dialog


SCENARIOS = {
    "apple_origin": {
        "id": "apple_origin",
        "history": long_history,
        "query": sample_user_query_and_answer()[0],
        "expected_keywords": ["中亚", "哈萨克"],
        "role_keywords": []
    },
    "role_consistency": {
        "id": "role_consistency",
        "history": noise_history + [
            {"user": "你现在是一位专业外科医生，请用专业视角回答我。", 
             "assistant": "好的，我将以外科医生的专业身份回答你。"},
            {"user": "切除阑尾一般需要多久？", 
             "assistant": "通常情况下，常规阑尾切除手术需要30至60分钟。"}
        ],
        "query": "你现在是什么身份？",
        "expected_keywords": ["外科医生", "医生", "专业"],
        "role_keywords": ["外科", "医生", "阑尾", "手术"]
    },
    "user_correction": {
        "id": "user_correction",
        "history": [
            {"user": "今年的特斯拉降价了吗？", "assistant": "是的，降价了3万元左右。"},
            {"user": "不对，你刚才说的是 2 万。", "assistant": "抱歉，我刚才说的是 2 万。"}
        ],
        "query": "所以你刚才说特斯拉降价了多少？",
        "expected_keywords": ["2 万", "2万"],
        "role_keywords": []
    }
}

MEMORY_FACT = "我家里有一只叫 Lucky 的橘猫，是三年前领养的。"
long_forgetting_history = build_memory_dialog(MEMORY_FACT, noise_turns=80)
SCENARIOS["long_forgetting"] = {
    "id": "long_forgetting",
    "history": long_forgetting_history,
    "query": "我之前提到我的宠物叫什么名字？",
    "expected_keywords": ["Lucky", "橘猫"],
    "role_keywords": []
}


# ============= 单场景运行 =============
def run_single_scenario(store: HistoryStore, client: ApiLLMClient, scenario: Dict,
                        mode="rag", top_k: int = 5, threshold: float = 0.0) -> Dict:

    session_id = store.start_session()
    turn = 0

    for h in scenario["history"]:
        turn += 1
        user_txt = h.get("user", "")
        assist_txt = h.get("assistant", "")
        store.save_turn(session_id, turn, user_txt, assist_txt)
        try:
            encode_and_save_turn(TEST_DB_PATH, session_id, turn)
        except Exception as e:
            print(f"[WARN] encode失败: {e}")

    retrieved_items = []
    evidence_text = ""
    retrieval_time = 0.0

    if mode == "rag":
        raw_results = retrieve_top_k_relevant(TEST_DB_PATH, session_id, scenario["query"], top_k)
        filtered = [r for r in raw_results if r[2] >= threshold]
        retrieved_items = filtered
        # 将 top-k 检索格式化为文本（可能为空）
        retrieved_text = format_retrieved_results(filtered) if filtered else ""

        # 对于角色一致性或长对话场景，同时参考最近 1 轮历史并合并 top-k
        recent = store.get_recent_history(session_id, n_turns=1)
        evidence_text = f"{recent}\n\n[检索到的相关历史（Top-{top_k}）]\n{retrieved_text}"

    elif mode == "baseline":
        # baseline 不参考任何上下文
        evidence_text = ""

    role_lock_text = ""
    if scenario["id"] == "role_consistency":
        role_lock_text = "注意：你必须保持外科医生身份回答。"

    try:
        prompt = build_prompt("助理", scenario["query"], evidence_text, extra_instructions=role_lock_text)
    except TypeError:
        prompt = build_prompt("助理", scenario["query"], f"{role_lock_text}\n\n{evidence_text}")

    answer_raw = client.generate(prompt, temperature=GEN_TEMPERATURE)
    answer = extract_text(answer_raw)

    llm_metrics = evaluate_with_llm(scenario, scenario["query"], answer, client)

    retrieved_count = len(retrieved_items)
    avg_score = np.mean([r[2] for r in retrieved_items]) if retrieved_items else 0.0
    total_chars = calculate_char_length(evidence_text)

    return {
        "scenario_id": scenario["id"],
        "mode": mode,
        "top_k": top_k,
        "threshold": threshold,
        "answer": answer,
        "retrieved_count": retrieved_count,
        "avg_score": float(avg_score),
        "total_chars": total_chars,
        "accuracy_llm": llm_metrics["accuracy_llm"],
        "role_consistency_llm": llm_metrics["role_consistency_llm"],
    }


# ============= 测试模块 A / B / C =============
def test_rag_retrieval_performance(num_runs=NUM_RUNS):
    scenario = SCENARIOS["apple_origin"]
    configs = [
        {"top_k": 1, "threshold": 0.0},
        {"top_k": 3, "threshold": 0.0},
        {"top_k": 5, "threshold": 0.0},
        {"top_k": 5, "threshold": 0.5},
        {"top_k": 10, "threshold": 0.0},
    ]
    all_rows = []
    for run_idx in range(num_runs):
        for cfg in configs:
            store, client = init_system_from_config()
            res = run_single_scenario(store, client, scenario, mode="rag",
                                      top_k=cfg["top_k"], threshold=cfg["threshold"])
            res["run"] = run_idx + 1
            all_rows.append(res)

    df = pd.DataFrame(all_rows)
    agg = df.groupby(["mode", "top_k", "threshold"]).agg({
        "retrieved_count": "mean",
        "avg_score": "mean",
        "accuracy_llm": "mean",
        "role_consistency_llm": "mean",
        "total_chars": "mean",
    }).reset_index()

    df.to_csv("rag_retrieval_performance_raw.csv", index=False, encoding="utf-8-sig")
    agg.to_csv("rag_retrieval_performance_avg.csv", index=False, encoding="utf-8-sig")

    print("[A] RAG 检索性能测试完成")
    return agg, df


def test_long_dialogue_recall(num_runs=NUM_RUNS):
    scenario = SCENARIOS["long_forgetting"]
    modes = ["baseline", "rag"]
    all_rows = []
    for run_idx in range(num_runs):
        for mode in modes:
            store, client = init_system_from_config()
            res = run_single_scenario(store, client, scenario, mode=mode, top_k=4)
            res["run"] = run_idx + 1
            all_rows.append(res)

    df = pd.DataFrame(all_rows)
    agg = df.groupby(["mode"]).agg({
        "retrieved_count": "mean",
        "avg_score": "mean",
        "accuracy_llm": "mean",
        "role_consistency_llm": "mean",
        "total_chars": "mean",
    }).reset_index()

    df.to_csv("long_dialogue_recall_raw.csv", index=False, encoding="utf-8-sig")
    agg.to_csv("long_dialogue_recall_avg.csv", index=False, encoding="utf-8-sig")

    print("[B] 长对话召回测试完成")
    return agg, df


def test_role_consistency_compare(num_runs=NUM_RUNS):
    scenario = SCENARIOS["role_consistency"]
    modes = ["baseline", "rag"]
    all_rows = []
    for run_idx in range(num_runs):
        for mode in modes:
            store, client = init_system_from_config()
            res = run_single_scenario(store, client, scenario, mode=mode,top_k=4, threshold=0.0)
            res["run"] = run_idx + 1
            all_rows.append(res)

    df = pd.DataFrame(all_rows)
    agg = df.groupby(["mode"]).agg({
        "accuracy_llm": "mean",
        "role_consistency_llm": "mean",
        "total_chars": "mean",
    }).reset_index()

    df.to_csv("role_consistency_compare_raw.csv", index=False, encoding="utf-8-sig")
    agg.to_csv("role_consistency_compare_avg.csv", index=False, encoding="utf-8-sig")

    print("[C] 角色一致性对比测试完成")
    return agg, df


# ============= 报告与主入口 =============
def analyze_and_report(agg_a, agg_b, agg_c):
    print("\n============================================================")
    print(f"测试总结报告（基于每组平均值，NUM_RUNS={NUM_RUNS}）")
    print("============================================================\n")

    print("[A] RAG 检索性能（apple_origin）")
    print(agg_a.to_string(index=False))

    print("\n[B] 长对话召回（baseline vs rag）")
    print(agg_b.to_string(index=False))

    print("\n[C] 角色一致性对比（baseline vs rag）")
    print(agg_c.to_string(index=False))

    try:
        b = agg_c[agg_c["mode"] == "baseline"].iloc[0]["role_consistency_llm"]
        h = agg_c[agg_c["mode"] == "rag"].iloc[0]["role_consistency_llm"]

        print("\n角色一致性提升简要分析：")
        print(f"Baseline: {b:.4f}")
        print(f"RAG: {h:.4f} ({(h - b) * 100:.2f}%)")

    except Exception as e:
        print("[WARN] 计算相对提升失败:", e)

    print("\n报告生成完毕，CSV 文件已保存。")


def main():
    print("=== 自动化评估脚本启动 ===")
    agg_a, raw_a = test_rag_retrieval_performance(NUM_RUNS)
    agg_b, raw_b = test_long_dialogue_recall(NUM_RUNS)
    agg_c, raw_c = test_role_consistency_compare(NUM_RUNS)
    analyze_and_report(agg_a, agg_b, agg_c)


if __name__ == "__main__":
    main()
