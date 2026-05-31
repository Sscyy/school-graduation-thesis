"""可视化工具：将推理结果生成为自包含的 HTML 对比报告。

每次 infer 生成一个 viz.html 文件，50 行 × 3 列（原始图 | GT | 预测），
图像以 Base64 内嵌，HTML 无需外部依赖，便于跨实验横向对比。
"""

import base64

import cv2
import numpy as np


def _img_to_base64(img_rgb: np.ndarray) -> str:
    """将 RGB uint8 图像编码为 Base64 PNG 字符串（用于 HTML 内嵌）。"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf   = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf).decode('utf-8')


def save_viz_html(
    viz_data: list,
    output_path: str,
    title: str = "Inference Visualization",
):
    """生成自包含的 HTML 可视化对比报告。

    Parameters
    ----------
    viz_data    : list of (orig, gt, pred)
                  每个元素均为 (H, W, 3) uint8 **RGB** numpy array
    output_path : HTML 文件保存路径（含文件名，如 .../infer/viz.html）
    title       : 页面标题，建议写实验名，如 "A3: SAM2.1 + BBE"
    """
    rows_html = []
    for i, (orig, gt, pred) in enumerate(viz_data, start=1):
        b64_orig = _img_to_base64(orig)
        b64_gt   = _img_to_base64(gt)
        b64_pred = _img_to_base64(pred)
        rows_html.append(f"""    <tr>
      <td class="idx">{i}</td>
      <td><img src="data:image/png;base64,{b64_orig}" alt="orig"></td>
      <td><img src="data:image/png;base64,{b64_gt}"   alt="gt"></td>
      <td><img src="data:image/png;base64,{b64_pred}" alt="pred"></td>
    </tr>""")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body  {{ font-family: "Segoe UI", Arial, sans-serif; margin: 28px;
             background: #f2f3f5; color: #333; }}
    h2    {{ margin-bottom: 14px; font-size: 20px; color: #1a1a2e; }}
    table {{ border-collapse: collapse; background: #fff;
             box-shadow: 0 2px 8px rgba(0,0,0,.10); border-radius: 6px;
             overflow: hidden; }}
    thead th {{ background: #2e5bba; color: #fff; padding: 11px 20px;
                font-size: 14px; letter-spacing: .4px; }}
    tbody td {{ padding: 7px 10px; text-align: center;
                border-bottom: 1px solid #ebebeb; }}
    td.idx    {{ background: #f7f7f7; color: #999; font-size: 12px;
                 font-weight: 600; width: 36px; }}
    img       {{ width: 164px; height: 164px; display: block; margin: auto;
                 image-rendering: pixelated; border-radius: 3px; }}
    tbody tr:hover td      {{ background: #eef4ff; }}
    tbody tr:hover td.idx  {{ background: #ddeaff; }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>原始图像</th>
        <th>真实标注（GT）</th>
        <th>模型预测</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(rows_html)}
    </tbody>
  </table>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Visualisation HTML saved: {output_path}  ({len(viz_data)} examples)")
