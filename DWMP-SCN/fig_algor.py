import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')  # 关闭坐标轴

    # --- 辅助函数：画圆角矩形 ---
    def draw_box(x, y, w, h, text, color='#e6f2ff', edge='black'):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                     linewidth=1.5, edgecolor=edge, facecolor=color)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=11, fontweight='bold', wrap=True)
        return x + w/2, y, y + h  # 返回中心x, 下边缘y, 上边缘y

    # --- 辅助函数：画箭头 ---
    def draw_arrow(x1, y1, x2, y2, text=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
        if text:
            ax.text((x1+x2)/2 + 0.1, (y1+y2)/2, text, fontsize=10, color='#333333')

    # ==========================================
    # 区域 1: 离线训练阶段 (Phase 1)
    # ==========================================
    # 画大虚线框背景
    rect1 = patches.Rectangle((0.5, 7.5), 9, 6, linewidth=1.5, edgecolor='#1f77b4', 
                              facecolor='none', linestyle='--', alpha=0.5)
    ax.add_patch(rect1)
    ax.text(1.0, 13.2, "Phase 1: Model Pool Construction (Offline)", 
            fontsize=12, fontweight='bold', color='#1f77b4')

    # 1. 输入数据
    cx, by, ty = draw_box(3.5, 12.0, 3, 0.8, "Training Data (CCPP)", color='#fff2cc')
    
    # 2. 循环构建
    cx_loop, by_loop, ty_loop = draw_box(2.0, 10.0, 6, 1.2, 
                                         "Loop M times (e.g., 50):\n1. Bagging (80% Data)\n2. Weak Reg SCN Training", 
                                         color='#e6f2ff')
    draw_arrow(cx, by, cx, ty_loop)

    # 3. 筛选
    cx_sel, by_sel, ty_sel = draw_box(3.0, 8.2, 4, 1.0, 
                                      "Select Top-K Models\n(Based on RMSE)", 
                                      color='#d9ead3')
    draw_arrow(cx_loop, by_loop, cx_sel, ty_sel)

    # ==========================================
    # 区域 2: 在线预测阶段 (Phase 2)
    # ==========================================
    # 画大虚线框背景
    rect2 = patches.Rectangle((0.5, 0.5), 9, 6.5, linewidth=1.5, edgecolor='#d62728', 
                              facecolor='none', linestyle='--', alpha=0.5)
    ax.add_patch(rect2)
    ax.text(1.0, 6.7, "Phase 2: Dynamic Prediction (Online)", 
            fontsize=12, fontweight='bold', color='#d62728')

    # 4. 新数据输入
    cx_new, by_new, ty_new = draw_box(0.8, 5.5, 2.5, 0.8, "New Input x(t)", color='#fff2cc')
    
    # 5. 模型池
    cx_pool, by_pool, ty_pool = draw_box(3.5, 5.0, 3, 1.5, 
                                         "Model Pool\n{M1, M2, ..., M10}", 
                                         color='#e1d5e7')
    # 连接 Phase 1 到 Phase 2
    draw_arrow(cx_sel, by_sel, cx_pool, ty_pool, text="Save Pool")
    draw_arrow(cx_new + 1.25, 5.9, cx_pool, 5.9) # 输入到池

    # 6. Softmax 加权
    cx_weight, by_weight, ty_weight = draw_box(3.0, 3.0, 4, 1.2, 
                                               "Dynamic Weighting (Softmax)\nLambda = 80\nBased on error e(t-1)", 
                                               color='#f4cccc')
    draw_arrow(cx_pool, by_pool, cx_weight, ty_weight, text="Sub-predictions")

    # 7. 最终输出
    cx_out, by_out, ty_out = draw_box(3.5, 1.0, 3, 0.8, "Final Output y(t)", color='#d9ead3')
    draw_arrow(cx_weight, by_weight, cx_out, ty_out, text="Weighted Sum")

    # 8. 反馈回路 (Feedback Loop) - 体现 "Dynamic"
    # 画一条折线箭头
    ax.annotate("", xy=(cx_weight + 2.0, 3.6), xytext=(cx_out + 1.5, 1.4),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#d62728', connectionstyle="bar,angle=180,fraction=-0.4"))
    ax.text(7.2, 2.5, "Update Error\ne(t-1)", fontsize=10, color='#d62728', ha='center')

    plt.tight_layout()
    plt.show() # 保存图片

draw_flowchart()