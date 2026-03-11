import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


# =============================================================
# 字体配置：强制全局使用 Times New Roman
# =============================================================
def set_times_new_roman():
    """全局设置 Times New Roman 字体并返回 FontProperties"""
    font_path = r'C:\Windows\Fonts\times.ttf'  # 确认系统中该路径存在
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Times New Roman 字体文件未找到: {font_path}")

    font_prop = font_manager.FontProperties(fname=font_path)
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.unicode_minus': False,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })
    return font_prop


font_prop = set_times_new_roman()

# =============================================================
# 输入/输出路径与配色方案
# =============================================================
INPUT_PERIOD_FILES = {
    "2021-2040": "ALL-RATE_N_126_2021-2040.csv",
    "2041-2060": "ALL-RATE_N_126_2041-2060.csv",
    "2061-2080": "ALL-RATE_N_126_2061-2080.csv",
    "2081-2100": "ALL-RATE_N_126_2081-2100.csv",
}
YIELD_FILE = "nitrogen_y.csv"
OUTPUT_DIR = "results_opt_nc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCATTER_NORMAL_COLOR = "#AED6F1"  # 面板(a)浅蓝
SCATTER_ABNORMAL_COLOR = "#F9E79F"  # 面板(a)浅黄
BAR_COLORS = [
    "#D8BFD8",  # 浅紫
    "#FFD1DC",  # 浅粉
    "#B5C7D3",  # 灰蓝紫
    "#E0C0A0",  # 浅驼色
    "#A3B5A5",  # 灰绿
    "#C0B0A0",  # 浅褐
]


# =============================================================
# 绘图函数：双面板 + 精细布局 + 保证所有文字 New Roman
# =============================================================
def create_combined_plot(period, result_df, font_prop):
    # 建立子图，不用 constrained_layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 手动调整四周留白和面板间距
    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        top=0.88,
        bottom=0.12,
        wspace=0.30
    )

    # ---------------------------
    # (a) 成本 vs. 产量 散点图
    # ---------------------------
    x = result_df['Final_Cost']
    y = result_df['Selected_Value']
    mask = result_df['Is_Abnormal']
    ax1.scatter(x[~mask], y[~mask],
                s=50, color=SCATTER_NORMAL_COLOR,
                edgecolor='black', alpha=0.8, label='Normal (<=435)')
    ax1.scatter(x[mask], y[mask],
                s=70, marker='^', color=SCATTER_ABNORMAL_COLOR,
                edgecolor='black', alpha=0.9, label='Abnormal (>435)')

    # 强制 New Roman：标题、轴标签
    ax1.set_title('(a) Nitrogen Input vs. Selected Yield Value',
                  fontproperties=font_prop, fontsize=12)
    ax1.set_xlabel('Nitrogen input (kg/ha)',
                   fontproperties=font_prop, fontsize=10)
    ax1.set_ylabel('Selected Yield Value',
                   fontproperties=font_prop, fontsize=10)
    # 刻度标签
    for lbl in ax1.get_xticklabels() + ax1.get_yticklabels():
        lbl.set_fontproperties(font_prop)
    # 图例
    leg = ax1.legend(frameon=False)
    for txt in leg.get_texts():
        txt.set_fontproperties(font_prop)
    ax1.grid(True, linestyle=':')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ---------------------------
    # (b) 施肥比例分布 水平柱状图
    # ---------------------------
    ratio_pct = result_df['Fertilizer_Ratio'] \
                    .value_counts(normalize=True) \
                    .sort_index() * 100
    bars = ax2.barh(ratio_pct.index.astype(str),
                    ratio_pct.values,
                    height=0.6,
                    edgecolor='white',
                    color=BAR_COLORS[:len(ratio_pct)])
    # 添加百分比标签，保证在框内
    max_pct = ratio_pct.max()
    ax2.set_xlim(0, max_pct * 1.15)
    for bar in bars:
        w = bar.get_width()
        txt = ax2.text(bar.get_x() + w + 1,
                       bar.get_y() + bar.get_height() / 2,
                       f"{w:.1f}%",
                       va='center',
                       fontproperties=font_prop,
                       fontsize=9)
    # 强制 New Roman：标题、轴标签
    ax2.set_title('(b) Nitrogen Ratio Distribution (%)',
                  fontproperties=font_prop, fontsize=12)
    ax2.set_xlabel('Percentage (%)',
                   fontproperties=font_prop, fontsize=10)
    ax2.set_ylabel('Nitrogen Ratio',
                   fontproperties=font_prop, fontsize=10)
    # 刻度标签
    for lbl in ax2.get_xticklabels() + ax2.get_yticklabels():
        lbl.set_fontproperties(font_prop)
    ax2.grid(True, axis='y', linestyle=':')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    return fig


# =============================================================
# 主程序：载入数据、计算、绘图并保存
# =============================================================
if __name__ == "__main__":
    # 读取产量
    y_df = pd.read_csv(YIELD_FILE)
    y_array = y_df["wheat_Nitr"].values.reshape(-1, 1)

    for period, path in INPUT_PERIOD_FILES.items():
        print(f"处理时期：{period}")
        # 读取并标准化
        df = pd.read_csv(path, engine='python')
        mapping = {c: f"n{c.split('__')[0].split('_')[0][1:]}" for c in df.columns}
        df.rename(columns=mapping, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]

        # 计算效益比并选择
        props = np.array([int(c[1:]) / 100 for c in df.columns])
        vals = df.values
        costs = props * y_array
        ratio = vals / costs

        idx = np.argmax(ratio, axis=1)
        sel_vals = vals[np.arange(len(vals)), idx]
        sel_costs = costs[np.arange(len(vals)), idx]
        sel_props = props[idx]

        # 预算调整
        total_budget = y_array.sum()
        rem = total_budget - sel_costs.sum()
        for i in range(len(vals)):
            cur = idx[i]
            cur_val, cur_cost = vals[i, cur], costs[i, cur]
            for j in np.argsort(-vals[i]):
                if j == cur: continue
                alt_val, alt_cost = vals[i, j], costs[i, j]
                extra = alt_cost - cur_cost
                if (alt_cost <= 435 and
                        (extra <= rem or cur_cost > 435) and
                        (alt_val > cur_val or cur_cost > 435)):
                    idx[i] = j
                    sel_vals[i], sel_costs[i], sel_props[i] = alt_val, alt_cost, props[j]
                    rem -= max(extra, 0)
                    break

        # 汇总保存
        result_df = pd.DataFrame({
            'Selected_Column': df.columns[idx],
            'Selected_Value': sel_vals,
            'Fertilizer_Ratio': sel_props,
            'Y_Value': y_array.flatten(),
            'Final_Cost': sel_costs,
            'Is_Abnormal': sel_costs > 500
        })
        result_df.to_csv(os.path.join(OUTPUT_DIR, f"245_result_{period}.csv"), index=False)

        # 绘图、预览、保存
        fig = create_combined_plot(period, result_df, font_prop)
        plt.show(block=False)
        plt.pause(20)
        out_png = os.path.join(OUTPUT_DIR, f"{period}_245.png")
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"已保存图像：{out_png}\n")
