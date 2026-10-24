
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# 一、读取 Excel 数据
# 第一行：距离 d（mm）；第二行：角频率 omega（rad/s）。
# 第一列是标题；其余单元格可能含有 LaTeX 数学模式的 $ 符号。
data = pd.read_excel('扭摆的角频率.xlsx', header=None)
d = pd.to_numeric(data.iloc[0, 1:].astype(str).str.replace('$', '', regex=False), errors='coerce').to_numpy()
omega = pd.to_numeric(data.iloc[1, 1:].astype(str).str.replace('$', '', regex=False), errors='coerce').to_numpy()

# 二、整理为线性拟合坐标
# 原始 d 为 mm，除以 1000 转换为 m；omega 单位为 rad/s。
# 只保留有效的距离和正角频率，避免 1/omega² 除零。
mask = np.isfinite(d) & np.isfinite(omega) & (omega > 0)
x = (d[mask] / 1000) ** 2
y = 1 / omega[mask] ** 2

# 三、定义线性模型并用 SciPy 最小二乘拟合
# x = d²，y = 1/omega²，a 是斜率，b 是截距。
def function(x, a, b):
    return a * x + b

(a, b), covariance = curve_fit(function, x, y)

# 协方差矩阵对角线开平方得到参数的估计标准误差。
a_err, b_err = np.sqrt(np.diag(covariance))
y_fit = function(x, a, b)

# R² = 1 - 残差平方和 / 总离差平方和。
r2 = 1 - np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2)

# 四、创建 result 文件夹，输出 LaTeX 格式的文本报告
output = Path('results2')
output.mkdir(exist_ok=True)

# :.8g 表示保留 8 位有效数字；b:+.8g 自动附带正负号。
equation = rf'\frac{{1}}{{\omega^2}}={a:.8g}d^2{b:+.8g}'
latex = (
    '\\begin{align*}\n'
    + equation.replace('=', '&=', 1) + r'\\' + '\n'
    + rf'a &= ({a:.8g}\pm{a_err:.2g})\,\mathrm{{s^2/m^2}}\\' + '\n'
    + rf'b &= ({b:.8g}\pm{b_err:.2g})\,\mathrm{{s^2}}\\' + '\n'
    + rf'R^2 &= {r2:.8f}' + '\n'
    + '\\end{align*}\n'
)
(output / 'fit2.txt').write_text(latex, encoding='utf-8')

# 五、绘制散点及拟合直线，保存为矢量 PDF
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, label='Experimental data', zorder=3)
x_line = np.linspace(x.min(), x.max(), 200)
ax.plot(x_line, function(x_line, a, b), label='Linear fit')
ax.set_xlabel(r'$d^2\ (\mathrm{m^2})$')
ax.set_ylabel(r'$1/\omega^2\ (\mathrm{s^2})$')
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(output / 'fit.pdf')
plt.close(fig)

print('拟合完成：result/fit.pdf、result/fit.txt')
