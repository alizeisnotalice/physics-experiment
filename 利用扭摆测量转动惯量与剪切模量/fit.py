
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from pathlib import Path

# ==================================================
# 一、读取实验数据
# ==================================================

# CSV 每两列为一组：时间 t（s）和电压 U（V）
# utf-8-sig 兼容带 BOM 的 UTF-8 CSV 文件
data = pd.read_csv("data.csv", encoding="utf-8-sig")

# 创建结果文件夹
output = Path("results")
output.mkdir(exist_ok=True)


# ==================================================
# 二、定义阻尼振动函数
# ==================================================

# A：初始振幅（V）
# beta：振幅衰减系数（s^-1）
# omega：阻尼振动角频率（rad/s）
# phi：初相位（rad）
# C：电压偏移（V）
def function(t, A, beta, omega, phi, C):
    return A * np.exp(-beta * t) * np.cos(omega * t + phi) + C


# ==================================================
# 三、批量拟合
# ==================================================

# 每两列为一组，自动计算组数
for i in range(len(data.columns) // 2):

    # 提取当前组的时间、电压数据
    # 将非数值内容转换为 NaN，再删除缺失行
    group = data.iloc[:, 2*i:2*i+2].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    t = group.iloc[:, 0].to_numpy(dtype=float)
    U = group.iloc[:, 1].to_numpy(dtype=float)

    # 按时间升序排列，并删除重复时间点
    order = np.argsort(t)
    t, U = t[order], U[order]
    t, index = np.unique(t, return_index=True)
    U = U[index]

    # ==================================================
    # 四、自动估计初始参数
    # ==================================================

    # 电压平均值作为偏置初值
    C0 = np.mean(U)

    # 相邻采样间隔的中位数
    dt = np.median(np.diff(t))

    # FFT 估计振动角频率
    # rfft 计算实数信号的频谱
    # rfftfreq 生成对应频率坐标
    freq = np.fft.rfftfreq(len(t), d=dt)
    spectrum = np.abs(np.fft.rfft(U - C0))
    omega0 = 2 * np.pi * freq[np.argmax(spectrum[1:]) + 1]

    # Hilbert 变换提取振幅包络
    signal = hilbert(U - C0)
    envelope = np.abs(signal)

    # 只使用振幅较大的部分估计衰减系数
    mask = envelope > 0.1 * np.max(envelope)

    # ln(A) = ln(A0) - beta*t
    # 线性拟合斜率的相反数作为 beta 初值
    beta0 = max(
        0,
        -np.polyfit(t[mask], np.log(envelope[mask]), 1)[0]
    )

    # 初始振幅和初相位
    A0 = np.max(envelope)
    phi0 = np.angle(signal[0])

    # ==================================================
    # 五、SciPy 非线性最小二乘拟合
    # ==================================================

    # 尝试三个不同的频率初值
    # 选择残差平方和最小的拟合结果
    best = None

    for scale in [0.95, 1.0, 1.05]:
        try:
            popt, _ = curve_fit(
                function, t, U,
                p0=[A0, beta0, omega0 * scale, phi0, C0],
                bounds=(
                    [0, 0, 0, -np.inf, -np.inf],
                    [np.inf, np.inf, np.inf, np.inf, np.inf]
                ),
                maxfev=20000
            )

            # 计算残差平方和
            error = np.sum((U - function(t, *popt))**2)

            # 保存当前最优结果
            if best is None or error < best[0]:
                best = (error, popt)

        except (RuntimeError, ValueError):
            continue

    # 拟合失败时跳过当前组
    if best is None:
        print(f"第 {i+1} 组拟合失败")
        continue

    # ==================================================
    # 六、计算拟合参数和 R²
    # ==================================================

    # 取出五个拟合参数
    A, beta, omega, phi, C = best[1]

    # 计算拟合曲线
    U_fit = function(t, A, beta, omega, phi, C)

    # 计算决定系数
    SS_res = np.sum((U - U_fit)**2)
    SS_tot = np.sum((U - np.mean(U))**2)
    R2 = 1 - SS_res / SS_tot if SS_tot > 0 else np.nan

    # ==================================================
    # 七、生成 LaTeX 格式 TXT
    # ==================================================

    # 将参数代入完整方程
    # :.8g 表示保留八位有效数字
    # :+ 表示正数带 +，负数带 -
    equation = (
        rf"U(t)&={A:.8g}e^{{-{beta:.8g}t}}"
        rf"\cos\left({omega:.8g}t{phi:+.8g}\right)"
        rf"{C:+.8g}"
    )

    # 生成完整 LaTeX 内容
    # align* 环境对齐等号，不显示公式编号
    latex = (
        "\\begin{align*}\n"
        + equation + r"\\" + "\n"
        + rf"A &= {A:.8g}\,\mathrm{{V}}\\" + "\n"
        + rf"\beta &= {beta:.8g}\,\mathrm{{s^{{-1}}}}\\" + "\n"
        + rf"\omega &= {omega:.8g}\,\mathrm{{rad/s}}\\" + "\n"
        + rf"\varphi &= {phi:.8g}\,\mathrm{{rad}}\\" + "\n"
        + rf"C &= {C:.8g}\,\mathrm{{V}}\\" + "\n"
        + rf"R^2 &= {R2:.8f}" + "\n"
        + "\\end{align*}\n"
    )

    # 每组保存一个 TXT 文件
    with open(
        output / f"group_{i+1:02d}.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(latex)

    # ==================================================
    # 八、绘制 PDF
    # ==================================================

    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 6))

    # 实验曲线与拟合曲线
    ax.plot(t, U, label="Experimental data", linewidth=0.7)
    ax.plot(t, U_fit, label="Fitted curve", linewidth=1.2)

    # 坐标轴与图例
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()

    # 自动调整布局
    fig.tight_layout()

    # 保存 PDF，不显示拟合方程和参数
    fig.savefig(output / f"group_{i+1:02d}.pdf")

    # 关闭图像，避免占用过多内存
    plt.close(fig)

    print(f"第 {i+1} 组完成，R² = {R2:.8f}")
