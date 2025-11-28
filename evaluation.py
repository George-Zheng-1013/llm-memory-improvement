import re
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def analyze_rag_logs(log_dir="results/a"):
    """
    分析指定目录下的 RAG 日志文件，计算 T20 的召回率、精确率和字符消耗。
    """

    # 关键事实的关键词定义 (只要匹配其中一组关键词即视为召回)
    facts = {
        "F1 (狗/花生)": ["花生", "香蕉"],
        "F2 (餐厅/蓝月)": ["蓝色月亮", "湖边"],
        "F3 (车牌)": ["7421"],
        "F4 (日期)": ["2024", "5", "20"],
    }

    results = []

    # 查找所有 a_rag_*.txt 文件
    pattern = os.path.join(log_dir, "a_rag_*.txt")
    files = glob.glob(pattern)

    # 按 k 值排序 (文件名中提取数字)
    files.sort(key=lambda x: int(re.search(r"a_rag_(\d+).txt", x).group(1)))

    print(
        f"{'Top-K':<6} | {'Recall':<8} | {'Precision':<10} | {'Cost (Chars)':<12} | {'Score':<8} | {'Facts Found'}"
    )
    print("-" * 85)

    for file_path in files:
        try:
            k_val = int(re.search(r"a_rag_(\d+).txt", file_path).group(1))

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 提取最后一次“检索到的相关历史证据”块 (对应 T20)
            # 假设日志格式为： 【检索到的相关历史证据】 ... 【证据结束】
            evidence_blocks = re.findall(
                r"【检索到的相关历史证据】(.*?)【证据结束】", content, re.DOTALL
            )

            if not evidence_blocks:
                print(f"{k_val:<6} | No evidence block found.")
                continue

            # 取最后一个块 (T20)
            last_evidence = evidence_blocks[-1]

            # 计算字符消耗 (去除首尾空白)
            cost = len(last_evidence.strip())

            # 计算召回情况
            found_facts = []
            for fact_name, keywords in facts.items():
                # 简单的关键词匹配：如果 keywords 中的任意一个词出现在证据中，视为可能相关
                # 为了更严谨，这里要求 keywords 列表中的词都出现（如果定义了多个）
                # 但根据实际数据 F1="花生"+"香蕉"，F2="蓝色月亮" 比较稳妥
                if all(kw in last_evidence for kw in keywords):
                    found_facts.append(fact_name)

            recall_count = len(found_facts)
            total_facts = len(facts)
            recall = recall_count / total_facts

            # 计算精确率
            # 实际上 retrieved_count 应该等于 k，但在日志中我们可以数一下 "[证据#" 的数量
            retrieved_chunks = len(re.findall(r"\[证据#\d+\]", last_evidence))
            precision = recall_count / len(facts)

            # 计算评分函数 S = Wr * R + Wp * P - Wc * C/Cmax
            # Wr=0.5, Wp=0.3, Wc=0.2, Cmax=500
            Wr, Wp, Wc, Cmax = 0.5, 0.3, 0.2, 500
            score = Wr * recall + Wp * precision - Wc * (cost / Cmax)

            results.append(
                {
                    "k": k_val,
                    "recall": recall,
                    "precision": precision,
                    "cost": cost,
                    "score": score,
                    "found": found_facts,
                }
            )

            print(
                f"{k_val:<6} | {recall:.0%} ({recall_count}/{total_facts}) | {precision:.2f}       | {cost:<12} | {score:.4f} | {', '.join([f.split()[0] for f in found_facts])}"
            )

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # 找出得分最高的结果
    if results:
        best_result = max(results, key=lambda x: x["score"])
        print("\n最佳配置:")
        print(f"Top-K: {best_result['k']}, Score: {best_result['score']:.4f}")

    return results


if __name__ == "__main__":
    # 注意：请确保脚本运行路径正确，或者修改 log_dir 指向您的文件位置
    # 在此环境中，我们假设文件在当前路径或相对路径下
    print("开始分析 RAG 日志...\n")
    # 尝试当前目录或常见上传目录
    if os.path.exists("results/a"):
        analyze_rag_logs("results/a")
    else:
        # 如果找不到路径，尝试搜索当前目录
        analyze_rag_logs(".")

    plt.figure()
    plt.title("RAG Evaluation Scores")
    plt.xlabel("Top-K")
    plt.ylabel("Score")
    results = analyze_rag_logs("results/a")
    ks = [res["k"] for res in results]
    scores = [res["score"] for res in results]
    plt.plot(ks, scores, "o")
    f_cubic = interp1d(ks, scores, kind="cubic")
    ks_interp = np.linspace(min(ks), max(ks), 100)
    scores_interp = f_cubic(ks_interp)
    plt.plot(ks_interp, scores_interp, "-")
    plt.grid()
    plt.show()
