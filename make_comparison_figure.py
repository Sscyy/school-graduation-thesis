"""
make_comparison_figure.py

从 A1 / A3 推理 HTML 中自动挑选 3 个最具代表性的样本，
生成 4 列对比图（原始图像 | 真实标注 | A1 基线 | A3 本文），
输出为适合 LaTeX 论文的 PNG 文件。

用法：
    python make_comparison_figure.py \
        --a1_html /path/to/baseline.html \
        --a3_html /path/to/sam_A3.html \
        --output  /path/to/comparison.png \
        --n_rows  3
"""

import argparse
import base64
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 字体设置（macOS / Linux 中文支持）───────────────────────────────────────
plt.rcParams['font.family'] = ['Hiragino Sans GB', 'Heiti TC',
                                'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────────────────────────────────────
# HTML 解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_html_images(html_path: str):
    """解析 HTML 文件，返回 list of (orig_rgb, gt_rgb, pred_rgb) numpy 数组。"""
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'data:image/png;base64,([A-Za-z0-9+/=]+)'
    b64_list = re.findall(pattern, content)

    if len(b64_list) % 3 != 0:
        raise ValueError(f"{html_path}: 找到 {len(b64_list)} 张图，不是 3 的倍数")

    rows = []
    for i in range(len(b64_list) // 3):
        imgs = []
        for j in range(3):
            raw  = base64.b64decode(b64_list[i * 3 + j])
            arr  = np.frombuffer(raw, np.uint8)
            bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        rows.append(tuple(imgs))   # (orig, gt, pred) — 全部 RGB

    print(f"  {html_path}: 解析到 {len(rows)} 行")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 自动选行：挑 A3 相比 A1 改善最明显的 n 行
# ─────────────────────────────────────────────────────────────────────────────

def _contour_mask(img_rgb: np.ndarray, color: str) -> np.ndarray:
    """从叠加了轮廓的 RGB 图中提取轮廓像素掩码。

    橙色（预测轮廓）：R > 200, G in [80,180], B < 80
    绿色（GT 轮廓） ：R < 80,  G > 200,    B < 80
    """
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    if color == 'orange':
        return (r > 200) & (g > 80) & (g < 180) & (b < 80)
    elif color == 'green':
        return (r < 80) & (g > 200) & (b < 80)
    raise ValueError(color)


def _overlap_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """预测轮廓与 GT 轮廓的 Dice 相似度（越大越好）。"""
    inter = (pred_mask & gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    return float(2 * inter / (denom + 1e-6))


def select_best_rows(a1_rows, a3_rows, n: int = 3):
    """为每行计算 A3-A1 的改善量，选出改善最大且分布均匀的 n 行。

    评分 = A3_轮廓-GT 重叠度  −  A1_轮廓-GT 重叠度
    分布均匀：从前 15 名里按等间距取 n 个，避免全选相邻样本。
    """
    assert len(a1_rows) == len(a3_rows), "A1/A3 行数不匹配"

    scores = []
    for i, ((_, gt, a1_pred), (_, _, a3_pred)) in enumerate(zip(a1_rows, a3_rows)):
        gt_mask   = _contour_mask(gt,      'green')
        a1_mask   = _contour_mask(a1_pred, 'orange')
        a3_mask   = _contour_mask(a3_pred, 'orange')

        a1_score  = _overlap_score(a1_mask, gt_mask)
        a3_score  = _overlap_score(a3_mask, gt_mask)
        improvement = a3_score - a1_score

        # 辅助条件：GT 要有足够多轮廓（剔除空白图）
        if gt_mask.sum() > 200:
            scores.append((improvement, i))

    # 降序排列，取前 15
    scores.sort(reverse=True)
    pool = [idx for _, idx in scores[:15]]

    # 在 pool 里等间距取 n 个（保证视觉多样性）
    if len(pool) <= n:
        selected = pool
    else:
        step = len(pool) / n
        selected = [pool[int(k * step)] for k in range(n)]

    selected = sorted(selected)   # 保持行号从小到大
    top_improvements = [scores[i][0] for i in range(min(5, len(scores)))]
    print(f"  Top-5 改善分数: {[f'{v:.4f}' for v in top_improvements]}")
    print(f"  选中行（1-indexed）: {[s + 1 for s in selected]}")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# 生成对比图
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_figure(a1_rows, a3_rows, selected_indices, output_path: str):
    """绘制 n×4 对比图并保存。"""
    n = len(selected_indices)

    col_labels = ['原始图像', '真实标注（GT）', 'A1 基线\n(ResNet50)', 'A3 本文\n(SAM 2.1+BBE)']
    row_labels  = [f'样本 {i + 1}' for i in range(n)]

    fig, axes = plt.subplots(
        n, 4,
        figsize=(16, n * 3.6 + 0.6),
        gridspec_kw={'hspace': 0.06, 'wspace': 0.04},
    )
    if n == 1:
        axes = [axes]   # 统一为二维列表

    # 列标题
    for col, label in enumerate(col_labels):
        axes[0][col].set_title(label, fontsize=13, fontweight='bold', pad=8,
                               color='#1a1a2e')

    for row_i, idx in enumerate(selected_indices):
        orig, gt, a1_pred = a1_rows[idx]
        _,    _,  a3_pred = a3_rows[idx]

        imgs = [orig, gt, a1_pred, a3_pred]
        for col, img in enumerate(imgs):
            ax = axes[row_i][col]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            # 边框颜色：A3 列用蓝色高亮，其余灰色
            color = '#2e5bba' if col == 3 else '#cccccc'
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.8 if col == 3 else 0.8)

        # 行标签（左侧）
        axes[row_i][0].set_ylabel(row_labels[row_i], fontsize=11,
                                   rotation=0, labelpad=48,
                                   va='center', color='#555')

    # 图例
    legend_handles = [
        mpatches.Patch(color=(0, 1, 0), label='真实标注轮廓（绿）'),
        mpatches.Patch(color=(1, 0.5, 0), label='模型预测轮廓（橙）'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  图像已保存: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="生成 A1 vs A3 定性对比图")
    p.add_argument("--a1_html", required=True)
    p.add_argument("--a3_html", required=True)
    p.add_argument("--output",  default="comparison.png")
    p.add_argument("--n_rows",  type=int, default=3)
    args = p.parse_args()

    print("── 解析 HTML ──────────────────────────────────────")
    a1_rows = parse_html_images(args.a1_html)
    a3_rows = parse_html_images(args.a3_html)

    print("── 自动选行 ──────────────────────────────────────")
    selected = select_best_rows(a1_rows, a3_rows, n=args.n_rows)

    print("── 生成对比图 ────────────────────────────────────")
    make_comparison_figure(a1_rows, a3_rows, selected, args.output)

    print("── 完成 ─────────────────────────────────────────")


if __name__ == "__main__":
    main()
