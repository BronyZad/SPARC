import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取三张表
df_lb = pd.read_csv("Final_LongBench.csv")
df_le = pd.read_csv("Final_LEval.csv")
df_ru = pd.read_csv("Final_RULER.csv")

methods = ["Native-Baseline", "Uniform-INT4", "Saber-BIC", "SnapKV"]

# 计算每个 Benchmark 的平均分
avg_data = []
for m in methods:
    avg_data.append({
        "Method": m,
        "LongBench (Avg)": pd.to_numeric(df_lb[m], errors='coerce').mean(),
        "L-Eval (Avg)": pd.to_numeric(df_le[m], errors='coerce').mean(),
        "RULER (Avg)": pd.to_numeric(df_ru[m], errors='coerce').mean()
    })

df_avg = pd.DataFrame(avg_data)
df_avg_melted = df_avg.melt(id_vars=["Method"], var_name="Benchmark", value_name="Average Score")

# 绘图
plt.figure(figsize=(10, 6))
sns.set_theme(style="ticks")

sns.barplot(data=df_avg_melted, x="Benchmark", y="Average Score", hue="Method", palette="viridis")

plt.title("Overall Macro-Average Performance across Benchmarks", fontsize=14, fontweight='bold')
plt.ylim(0, 100)
plt.legend(loc='upper left') # 图例放右下角比较空的地方
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("Plot_Overall_Average.png", dpi=300)
print("✅ 生成图片: Plot_Overall_Average.png")