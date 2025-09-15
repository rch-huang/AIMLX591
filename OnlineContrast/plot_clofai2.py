import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import csv
clofai_prefix = 'CDDB_seqcc_simclr_mem_'
def load_pad_to6(path, T_expected=None):
     
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # 跳过空行
            if not row:
                continue
            # 将非空单元转为 float
            vals = [float(x) for x in row if str(x).strip() != '']
            # 限制最多 6 个
            if len(vals) > 6:
                vals = vals[:6]
            # 填充到 6 个
            if len(vals) < 6:
                vals += [0.0] * (6 - len(vals))
            rows.append(vals)
    arr = np.array(rows, dtype=float)
    if T_expected is not None and arr.shape[0] != T_expected:
        raise ValueError(
            f"Row count mismatch: {path} has {arr.shape[0]} rows, expected {T_expected}"
        )
    return arr  # 形状 (T, 6)
# -------- 读取数据 --------
def load_csv(path):
    # 若有表头可改用 pandas.read_csv(header=None).values
    arr = np.loadtxt(path, delimiter=',')
    if arr.ndim == 1:  # 只有一行时保证二维
        arr = arr[None, :]
    return arr

cls = load_csv(clofai_prefix+'cls.csv')
import os
if os.path.exists(clofai_prefix+'cls_mem.csv'):
    cls_mem = load_csv(clofai_prefix+'cls_mem.csv')
else:
    cls_mem = None 
acc = load_csv(clofai_prefix+'knn_acc.csv')
n2 = 6  # 固定 6 条曲线（每组）

def preprocess_sum_into_col0_and_drop(arr, sum_cols=(0,2,4,6,8)):
    T, n = arr.shape
    # 只保留数据中存在的列索引
    valid_sum_cols = [c for c in sum_cols if c < n]
    if 0 not in valid_sum_cols:
        valid_sum_cols = [0] + [c for c in valid_sum_cols if c != 0]  # 确保0在里头以便写回

    # 需要删除的列（除0之外）
    drop_cols = [c for c in valid_sum_cols if c != 0 and c < n]

    # 计算“要写回0列”的总和
    summed = np.sum(arr[:, valid_sum_cols], axis=1)

    # 组装保留的列：保留0列 + 不在 drop_cols 里的其它列
    keep_cols = [0] + [c for c in range(1, n) if c not in drop_cols]

    out = arr[:, keep_cols].copy()
    # 0列的位置（一般是0，但写成通用）
    col0_pos = keep_cols.index(0)
    out[:, col0_pos] = summed
    return out, keep_cols

cls, kept_cols = preprocess_sum_into_col0_and_drop(cls, sum_cols=(0,2,4,6,8))
if cls_mem is not None:
    cls_mem, kept_cols2 = preprocess_sum_into_col0_and_drop(cls_mem, sum_cols=(0,2,4,6,8))

# 保险起见检查两边保留列一致（应该一致）
    if kept_cols != kept_cols2 or cls.shape != cls_mem.shape:
        raise RuntimeError("Preprocess mismatch between cls and cls_mem.")



T, n = cls.shape
x = np.arange(1, T + 1)  # x 从 1 开始（如果想从 0 开始，改成 np.arange(T)）
prec = load_pad_to6(clofai_prefix+'class_wise_precision.csv', T_expected=T)  # (T, 6)
rec  = load_pad_to6(clofai_prefix+'class_wise_recall.csv',    T_expected=T)  # (T, 6)
# -------- 颜色方案（n 种颜色）--------
# n<=20 用 tab20，更多时退回到 'hsv'
#cmap = plt.cm.get_cmap('tab20', n) if n <= 20 else plt.cm.get_cmap('hsv', n)
#colors = [cmap(i) for i in range(n)]
colors = ['r','g','b','c','m','y']
# -------- 画布与子图（先只画第一个）--------
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
ax = axs[0]

# cls：用点（无连线）
for j in range(n):
    ax.plot(
        x, cls[:, j],
        linestyle='dashed',  linewidth=1.2,
        color=colors[j], alpha=0.95
    )
if cls_mem is not None:
# cls_mem：用虚线
    for j in range(n):
        ax.plot(
            x, cls_mem[:, j],
            linestyle='dotted', linewidth=1.0,
            color=colors[j], alpha=0.95
        )
ax2 = axs[1]
# 颜色（为 6 种）
#cmap2 = plt.cm.get_cmap('tab10', n2)
colors2 = colors#[cmap2(i) for i in range(n2)]

eps = 0.0  # 若要把很小的数也当成0，可设为如1e-12

for j in range(n2):
    y_prec = prec[:, j]
    y_rec  = rec[:,  j]

    # 1) precision：先画线（无 marker）
    ax2.plot(
        x, y_prec,
        linestyle='-', linewidth=1.2,
        color=colors2[j], alpha=0.95
    )
    # 再只在非0处画点
    mask_p = ~np.isclose(y_prec, 0.0, atol=eps)
    ax2.plot(
        x[mask_p], y_prec[mask_p],
        linestyle='None', marker='.', markersize=4, markeredgewidth=0.8,
        color=colors2[j], alpha=0.95
    )

    # 2) recall：仅在非0处画叉（无连线）
    mask_r = ~np.isclose(y_rec, 0.0, atol=eps)
    ax2.plot(
        x[mask_r], y_rec[mask_r],
        linestyle='None', marker='x', markersize=8,
        color=colors2[j], alpha=0.95
    )
ax2.plot(x,acc[:,1],linestyle='dashdot', linewidth=3,
        color='b', alpha=0.95 )
ax2.set_title('Class-wise Precision (linedot) vs Recall (x)  And Averaged Accuracy (dashdot)')
ax2.set_ylabel('')
ax2.set_ylim(0.0, 1.0)

bin2 = load_csv(clofai_prefix+'binary_clustering.csv')  # 期望形状 (T, 2)
if bin2.shape[0] != T:
    raise ValueError(f"Row count mismatch: binary_clustering.csv has {bin2.shape[0]} rows, expected {T}")
# 兼容性处理：列不是 2 时进行截断/补零
if bin2.shape[1] < 2:
    pad = np.zeros((T, 2 - bin2.shape[1]), dtype=bin2.dtype)
    bin2 = np.hstack([bin2, pad])
elif bin2.shape[1] > 2:
    bin2 = bin2[:, :2]
ax3 = axs[2]
#cmap3 = plt.cm.get_cmap('tab10', 2)
colors3 = colors#[cmap3(0), cmap3(1)]
import matplotlib.colors as mcolors
ax3.plot(x, bin2[:, 0], linestyle='-',  linewidth=1.5, color='g', alpha=0.95, )
ax3.plot(x, bin2[:, 1], linestyle='-', linewidth=1.2, color='r', alpha=0.95, )

ax3.set_title('Real Images Detection Precision (green) vs Recall (red)')
ax3.set_ylabel('')
ax3.legend(loc='best', frameon=False)
# ax2.legend([
#     Line2D([0],[0], linestyle='-',  color='black', label='precision (solid)'),
#     Line2D([0],[0], linestyle='--', color='black', label='recall (dashed)'),
# ], loc='best', frameon=False)

# ====== ax3：占位 ======
axs[2].set_xlabel('x (row index)')
ax.set_title('Class distribution over tasks')
ax.set_ylabel('')

# 只给出图例说明两种风格，不逐列标注
legend_elements = [
    Line2D([0], [0], linestyle='dashed', label='Incoming Data', color='black'),
    Line2D([0], [0], linestyle='dotted', label='Memory Data', color='black'),
]




ax.legend(handles=legend_elements, loc='best', frameon=False)
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
for a in axs:
    a.set_xlim(1, T)
    # major：每 10 个；minor：每 1 个
    a.xaxis.set_major_locator(MultipleLocator(10))
    a.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    a.xaxis.set_minor_locator(MultipleLocator(1))
    # 调整刻度样式（可选）
    a.tick_params(axis='x', which='major', length=6)
    a.tick_params(axis='x', which='minor', length=3)
# 预留但不绘制第 2/3 子图（也可以注释掉这两行保留坐标轴）

axs[-1].set_xlabel('Sequence of Class-Incremetnal Tasks')

plt.tight_layout()

fig.savefig("clofai"+"_"+clofai_prefix+".jpg", format='jpg', dpi=300, bbox_inches='tight')