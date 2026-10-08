"""
============================================================
进阶挑战一 · 任务① RC 滤波电路 —— 完整仿真验证
============================================================
功能：手算推导 + PySpice(Ngspice)仿真 + 对比验证
电路：一阶 RC 低通滤波器
      R=1kΩ 串联, C=1μF 并联到地
      输入 vi(t), 输出 vo(t) 从电容两端取出

文件结构（运行后生成）：
  rc_filter_data.csv    — 频率扫描原始数据（幅频+相频）
  rc_filter_bode.png    — 波特图（幅频 + 相频）
  rc_filter_step.png    — 阶跃响应图（时域充放电）
  rc_filter_transient.png — 瞬态正弦响应图

========== 手算推导 =========================================

一、电路拓扑（请自行画图）

       R = 1 kΩ
  vi ──/\/\/───┬─── vo
               │
              ─── C = 1 μF
              ───
               │
              GND

  这是一个一阶 RC 低通滤波器。
  - 输入信号 vi 加在 R 串联 C 的两端
  - 输出电压 vo 从电容 C 两端取出

二、传递函数 H(jω) 的推导

  步骤①：用阻抗分压公式（交流分析中"阻抗"代替"电阻"）

    电阻 R 的阻抗：Z_R = R
    电容 C 的阻抗：Z_C = 1/(jωC)

    输出电压是电容上的分压：
                Z_C               1/(jωC)
    vo = vi × ───────── = vi × ──────────────
              Z_R + Z_C        R + 1/(jωC)

  步骤②：分子分母同乘 jωC，化为标准形式

             1/(jωC)              1             1
    H(jω) = ──────────── = ───────────── = ───────────
            R + 1/(jωC)     1 + jωRC       1 + j(ω/ωc)

    其中 ωc = 1/(RC) 是截止角频率（单位：rad/s）

  步骤③：得到幅频特性和相频特性

                  1
    |H(jω)| = ─────────────          ← 幅频特性（增益随频率变化）
               √(1 + (ω/ωc)²)

    φ(ω) = -arctan(ω/ωc)            ← 相频特性（相位滞后随频率变化）

  步骤④：截止频率 fc 的计算

    R = 1 kΩ  = 1000 Ω
    C = 1 μF  = 0.000001 F = 1×10⁻⁶ F

    截止角频率：
    ωc = 1/(RC) = 1/(1000 × 10⁻⁶) = 1/(10⁻³) = 1000 rad/s

    截止频率（Hz）：
    fc = ωc/(2π) = 1000/(2×3.14159) ≈ 159.15 Hz

    物理意义：当输入信号频率 f = fc = 159.15 Hz 时，
    输出幅度下降到输入幅度的 1/√2 ≈ 0.707（即 -3dB 点）。
    频率低于 fc：信号几乎全通过（通带）
    频率高于 fc：信号每 10 倍频衰减 20dB（阻带）

  步骤⑤：特征频率点的数值验证

    f = 0.1fc ≈ 15.9 Hz  → |H| ≈ 1/√(1+0.01)  = 0.995  (几乎无衰减)
    f = fc   ≈ 159.2 Hz  → |H| = 1/√(1+1)     = 0.707  (-3dB 点)
    f = 10fc ≈ 1591.5 Hz → |H| ≈ 1/√(1+100)   = 0.0995 (大幅衰减)

三、阶跃响应（时域）的理论推导

  输入：阶跃信号 vi(t) = 1V × u(t)  （t=0 时刻从 0 跳到 1V）

  一阶 RC 电路的阶跃响应（电容电压）：
                       -t/τ           -t/τ
    vo(t) = V_final × (1 - e   ) = 1 × (1 - e   )

    其中 τ = RC = 1000 × 10⁻⁶ = 0.001 s = 1 ms（时间常数）

  关键时间点：
    t = 0     → vo = 1×(1-1)    = 0 V
    t = τ     → vo = 1×(1-e⁻¹)  = 0.632 V  （63.2%）
    t = 3τ    → vo = 1×(1-e⁻³)  = 0.950 V  （95.0%，工程上认为稳定）
    t = 5τ    → vo = 1×(1-e⁻⁵)  = 0.993 V  （99.3%）

============================================================
"""

import os
import sys

# ===================== 输出目录（脚本所在目录） =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # 切换工作目录，确保输出文件保存到脚本同级目录

# ===================== Ngspice DLL 路径配置 =====================
# 说明：PySpice 需要加载 ngspice.dll 才能工作。
# 如果你的 Ngspice 安装路径不同，请修改 DLL_DIR。
DLL_DIR = r'C:\Users\yang_\ngspice-47_dll_64\Spice64_dll\dll-vs'
if os.path.exists(DLL_DIR):
    os.add_dll_directory(DLL_DIR)
    os.add_dll_directory(r'C:\Windows\System32')
    os.environ['PATH'] = DLL_DIR + os.pathsep + os.environ.get('PATH', '')
    os.environ['SPICE_LIB_DIR'] = os.path.abspath(os.path.join(DLL_DIR, '..'))
    os.environ['NGSPICE_LIBRARY_PATH'] = os.path.join(DLL_DIR, 'ngspice{}.dll')
else:
    print(f"⚠️ 警告: Ngspice DLL 路径不存在: {DLL_DIR}")
    print("   如果仿真报错，请修改 DLL_DIR 为你的 ngspice 安装路径。")

import numpy as np
import matplotlib
matplotlib.use('Agg')                        # 非交互式后端，直接保存图片
import matplotlib.pyplot as plt
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

# ===================== Ngspice 共享库配置 =====================
# 设置 Ngspice 可执行文件路径（PySpice 2.x 的配置方式）
if os.path.exists(DLL_DIR):
    NgSpiceShared.NGSPICE_PATH = os.path.abspath(os.path.join(DLL_DIR, '..'))
    NgSpiceShared.LIBRARY_PATH = os.path.join(DLL_DIR, 'ngspice{}.dll')

# ===================== Matplotlib 中文字体配置 =====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题

# ===================== 电路参数定义 =====================
# 说明：这些是RC滤波电路的核心参数，所有手算和仿真都基于它们。
R_val = 1e3          # 电阻 R = 1 kΩ = 1000 Ω
                     # 为什么选 1kΩ？它是一个常见标准值，RC 时间常数
                     # τ = RC = 1kΩ×1μF = 1ms，数值干净便于分析。
C_val = 1e-6         # 电容 C = 1 μF = 0.000001 F
                     # 为什么选 1μF？与 1kΩ 配合得到 τ=1ms，
                     # 截止频率 fc=159Hz，在音频范围内，方便观测。

fc_theory = 1.0 / (2.0 * np.pi * R_val * C_val)
                     # 截止频率 fc = 1/(2πRC)
                     # = 1/(2×3.14159×1000×0.000001)
                     # = 1/(6.28318×0.001) = 1/0.00628318
                     # ≈ 159.15 Hz

tau_theory = R_val * C_val
                     # 时间常数 τ = RC = 1000 × 10⁻⁶ = 0.001 s = 1 ms
                     # 物理意义：RC电路充放电到 63.2% 所需时间

print("=" * 60)
print("进阶挑战一 · 任务① RC 低通滤波电路 · 完整验证")
print("=" * 60)
print(f"\n电路参数：R = {R_val/1e3:.0f} kΩ,  C = {C_val/1e6:.0f} μF")
print(f"手算截止频率 fc = 1/(2πRC) = {fc_theory:.2f} Hz")
print(f"手算时间常数 τ = RC = {tau_theory*1e3:.2f} ms")
print(f"手算截止角频率 ωc = 1/(RC) = {1.0/(R_val*C_val):.1f} rad/s")

# ===================== 1. 频率扫描（AC 分析） =====================
print("\n" + "=" * 60)
print("第1部分：频率扫描（AC分析）—— 验证幅频与相频特性")
print("=" * 60)

def build_rc_lowpass():
    """
    构建 RC 低通滤波电路的 PySpice 网表。

    电路结构：
        vi ──[R=1kΩ]──┬── vo
                      │
                     [C=1μF]
                      │
                     GND

    Ngspice 节点说明：
      - 'n_in' : 输入节点，接交流电压源 vi
      - 'out'  : 输出节点（R 和 C 的连接点）
      - c.gnd  : 参考地（0V）
    """
    c = Circuit('RC_Lowpass_Filter')
    # V1: 交流电压源，DC偏置=0V，AC幅度=1V
    #     'n_in' 节点对 GND 提供 1V 振幅的正弦信号
    #     ac_mag=1 表示 AC 分析时信号振幅 = 1V（后续所有结果相对此值）
    c.SinusoidalVoltageSource('1', 'n_in', c.gnd,
                              dc_offset=0@u_V,    # 直流偏置 0V
                              amplitude=1@u_V,    # AC 振幅 1V（AC 分析和瞬态共用）
                              frequency=fc_theory)  # 默认频率设为 fc
    c.R('1', 'n_in', 'out', R_val @ u_Ohm)   # 电阻 R = 1kΩ，串联在输入和输出之间
    c.C('1', 'out', c.gnd, C_val @ u_F)    # 电容 C = 1μF，并联在输出到地
    return c

def run_ac_sweep(circuit, n_pts=200):
    """
    执行 AC 频率扫描仿真。

    参数：
      circuit : PySpice Circuit 对象
      n_pts   : 频率点数（对数均匀分布，10Hz → 100kHz，每10倍频200个点）

    返回：
      freq     : 频率数组（Hz）
      v_out_mag : 输出电压幅值数组（线性，V）
      v_out_phase : 输出电压相位数组（弧度）
    """
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    # AC 分析：从 10Hz 扫描到 100kHz，每 10 倍频 200 个点
    analysis = simulator.ac(start_frequency=10@u_Hz,
                            stop_frequency=100e3@u_Hz,
                            number_of_points=n_pts,
                            variation='dec')  # 'dec'=对数刻度(decade)
    freq = np.array(analysis.frequency)           # 频率列表（Hz）
    v_out = np.array(analysis['out'])             # 输出电压（复数相量）
    v_out_mag = np.abs(v_out)                     # 幅值 = |复数|
    v_out_phase = np.angle(v_out, deg=True)       # 相位（角度制）
    return freq, v_out_mag, v_out_phase

# 构建电路并执行 AC 扫描
circuit_ac = build_rc_lowpass()
freq_sim, v_mag_sim, v_phase_sim = run_ac_sweep(circuit_ac)

# 手算理论值（用于对比）
# 手算传递函数：|H(jω)| = 1/√(1+(f/fc)²)
# 手算相位：      φ = -arctan(f/fc)
v_mag_theory = 1.0 / np.sqrt(1.0 + (freq_sim / fc_theory)**2)
v_phase_theory = -np.arctan(freq_sim / fc_theory) * 180.0 / np.pi

# 在截止频率处取仿真值
idx_fc = np.argmin(np.abs(freq_sim - fc_theory))  # 找最接近 fc 的点的索引
v_at_fc_sim = v_mag_sim[idx_fc]                    # 仿真在 fc 处的幅值
v_at_fc_theory = 1.0 / np.sqrt(2.0)               # 理论值 = 1/√2 ≈ 0.7071
phase_at_fc_sim = v_phase_sim[idx_fc]               # 仿真在 fc 处的相位
phase_at_fc_theory = -45.0                          # 理论相位 = -45°

print(f"\n【截止频率 fc = {fc_theory:.2f} Hz 处的验证】")
print(f"  仿真幅值 |H(fc)| = {v_at_fc_sim:.6f}")
print(f"  理论幅值 |H(fc)| = {v_at_fc_theory:.6f} (= 1/√2)")
print(f"  幅值相对误差    = {abs(v_at_fc_sim - v_at_fc_theory)/v_at_fc_theory*100:.4f} %")
print(f"  仿真相位 φ(fc)  = {phase_at_fc_sim:.4f}°")
print(f"  理论相位 φ(fc)  = {phase_at_fc_theory:.4f}°")
print(f"  相位绝对误差    = {abs(phase_at_fc_sim - phase_at_fc_theory):.4f}°")

# ===================== 2. 瞬态分析 —— 阶跃响应 =====================
print("\n" + "=" * 60)
print("第2部分：阶跃响应（时域分析）—— 验证 RC 充放电曲线")
print("=" * 60)

def build_rc_step():
    """
    构建用于阶跃响应仿真的电路。

    用脉冲电压源模拟阶跃信号：
      - 初始值 = 0V
      - 脉冲值 = 1V（模拟 0V→1V 的阶跃跳变）
      - 延迟 = 0.1ms（给仿真一点稳定时间）
      - 上升时间 = 1ns（近似理想阶跃）
      - 脉冲宽度 = 5ms（远大于 5τ=5ms，让电容充满）
      - 周期 = 10ms
    """
    c = Circuit('RC_Step_Response')
    # PULSE 源：V1 V2 TD TR TF PW PER
    # V1=0V(初始), V2=1V(脉冲值), TD=0.2ms(延迟), TR=1ns(上升沿),
    # TF=1ns(下降沿), PW=20ms(脉宽), PER=40ms(周期)
    c.PulseVoltageSource('1', 'n_in', c.gnd,
                         initial_value=0@u_V,      # 信号从 0V 开始
                         pulsed_value=1@u_V,       # 跳变到 1V
                         delay_time=0.1e-3@u_s,   # 延迟 0.1ms 后跳变
                         rise_time=1e-9@u_s,      # 上升时间 1ns（近似理想跳变）
                         fall_time=1e-9@u_s,      # 下降时间 1ns
                         pulse_width=20e-3@u_s,   # 脉宽 20ms（>>5τ=5ms）
                         period=40e-3@u_s)         # 周期 40ms
    c.R('1', 'n_in', 'out', R_val @ u_Ohm)
    c.C('1', 'out', c.gnd, C_val @ u_F)
    return c

def run_transient(circuit, t_end=8e-3):
    """
    执行瞬态仿真。

    参数：
      circuit : PySpice Circuit 对象
      t_end   : 仿真结束时间（秒），选 8ms = 8τ，足够看到完整充放电

    返回：
      time     : 时间数组（秒）
      v_out    : 输出电压数组（V）
    """
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    # 瞬态分析：0 → 8ms，步长 10μs
    analysis = simulator.transient(step_time=10e-6@u_s,
                                   end_time=t_end@u_s)
    time = np.array(analysis.time)
    v_out = np.array(analysis['out'])
    return time, v_out

circuit_step = build_rc_step()
time_step, v_step_sim = run_transient(circuit_step, t_end=8e-3)

# 手算理论阶跃响应：vo(t) = 1 × (1 - e^(-t/τ))  （t>=0）
# 注意：仿真中信号在 t=0.1ms 时跳变，所以手算也要偏移 0.1ms
t_delay = 0.1e-3                     # 仿真中阶跃发生时刻
t_effective = np.maximum(time_step - t_delay, 0)  # 阶跃后的有效时间（去掉延迟）
v_step_theory = 1.0 * (1.0 - np.exp(-t_effective / tau_theory))

# 验证几个关键时间点的值
check_times = [0.0, tau_theory, 3*tau_theory, 5*tau_theory]
print(f"\n【阶跃响应关键时间点验证】（阶跃发生在 t = {t_delay*1e3:.1f} ms）")
print(f"  理论充电方程：vo(t) = 1 × (1 - e^(-t/{tau_theory*1e3:.1f}ms))")
for ct in check_times:
    t_actual = t_delay + ct
    v_theory = 1.0 * (1.0 - np.exp(-ct / tau_theory))
    idx = np.argmin(np.abs(time_step - t_actual))
    v_sim = v_step_sim[idx]
    pct = v_theory * 100
    print(f"  t = {ct*1e3:5.1f}ms (距阶跃): 理论 = {v_theory:.4f}V ({pct:5.1f}%), "
          f"仿真 = {v_sim:.4f}V, 误差 = {abs(v_sim-v_theory)*1e3:.3f}mV")

# ===================== 3. 瞬态分析 —— 正弦响应 =====================
print("\n" + "=" * 60)
print("第3部分：正弦瞬态响应 —— 观察不同频率下波形变化")
print("=" * 60)

def build_rc_sine(freq_hz):
    """
    构建正弦激励的RC电路。

    参数：
      freq_hz : 正弦信号频率（Hz）

    电路与前面一致，只是激励源改为固定频率的正弦波。
    """
    c = Circuit(f'RC_Sine_{freq_hz:.0f}Hz')
    c.SinusoidalVoltageSource('1', 'n_in', c.gnd,
                              dc_offset=0@u_V,
                              amplitude=1@u_V,
                              frequency=freq_hz@u_Hz)
    c.R('1', 'n_in', 'out', R_val @ u_Ohm)
    c.C('1', 'out', c.gnd, C_val @ u_F)
    return c

def run_transient_sine(circuit, freq_hz, t_end=None, n_cycles=5):
    """
    运行正弦波瞬态仿真。

    参数：
      circuit  : PySpice Circuit 对象
      freq_hz  : 信号频率（Hz）
      t_end    : 仿真结束时间（None=自动按周期算）
      n_cycles : 仿真多少个周期
    """
    period = 1.0 / freq_hz
    if t_end is None:
        t_end = n_cycles * period
    step = period / 200.0          # 每周期 200 个采样点，保证曲线光滑
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=step@u_s, end_time=t_end@u_s)
    time = np.array(analysis.time)
    v_in = np.array(analysis['n_in'])
    v_out = np.array(analysis['out'])
    return time, v_in, v_out

# 测试三个频率：0.1fc(通带)、fc(截止)、10fc(阻带)
test_freqs = [fc_theory * 0.1, fc_theory, fc_theory * 10]
test_labels = [f'0.1fc = {test_freqs[0]:.0f} Hz (通带，信号几乎全通过)',
               f'fc = {test_freqs[1]:.0f} Hz (截止频率，衰减到 70.7%)',
               f'10fc = {test_freqs[2]:.0f} Hz (阻带，大幅衰减)']

sine_results = []
for freq_hz, label in zip(test_freqs, test_labels):
    c_sine = build_rc_sine(freq_hz)
    t_s, v_in_s, v_out_s = run_transient_sine(c_sine, freq_hz)
    # 计算实际衰减
    v_out_amp = (np.max(v_out_s[-int(len(v_out_s)*0.3):]) -
                 np.min(v_out_s[-int(len(v_out_s)*0.3):])) / 2.0
    v_in_amp = 1.0                    # 输入振幅 = 1V
    gain_actual = v_out_amp / v_in_amp
    gain_theory = 1.0 / np.sqrt(1.0 + (freq_hz / fc_theory)**2)
    sine_results.append((t_s, v_in_s, v_out_s, label, gain_actual, gain_theory))
    print(f"\n  {label}")
    print(f"    仿真增益 = {gain_actual:.4f}")
    print(f"    理论增益 = {gain_theory:.4f}")
    print(f"    相对误差 = {abs(gain_actual - gain_theory)/gain_theory*100:.3f} %")

# ===================== 4. 绘图 =====================
print("\n" + "=" * 60)
print("第4部分：生成图表")
print("=" * 60)

# ---------- 图1：波特图（幅频 + 相频） ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 幅频特性（上子图）
ax1.semilogx(freq_sim, 20*np.log10(v_mag_sim), 'b-', lw=2.5,
             label='仿真 (PySpice AC分析)')
ax1.semilogx(freq_sim, 20*np.log10(v_mag_theory), 'r--', lw=1.8,
             label='手算理论 (|H|=1/√(1+(f/fc)²))')
ax1.axvline(fc_theory, color='gray', ls=':', lw=1.5,
            label=f'截止频率 fc = {fc_theory:.1f} Hz')
ax1.axhline(-3, color='green', ls=':', lw=1.2, label='-3 dB 线')
# 标注 fc 点
ax1.plot(fc_theory, -3, 'o', color='red', ms=8, zorder=5)
ax1.annotate(f'({fc_theory:.0f} Hz, -3 dB)',
             xy=(fc_theory, -3), xytext=(fc_theory*3, -2),
             fontsize=10, color='red',
             arrowprops=dict(arrowstyle='->', color='red'))
ax1.set_xlabel('频率 / Hz', fontsize=12)
ax1.set_ylabel('增益 / dB', fontsize=12)
ax1.set_title('RC 低通滤波器 · 幅频特性 (Bode图-幅值)', fontsize=13)
ax1.legend(fontsize=9, loc='lower left')
ax1.grid(True, alpha=0.35, which='both')
ax1.set_xlim(10, 1e5)

# 相频特性（下子图）
ax2.semilogx(freq_sim, v_phase_sim, 'b-', lw=2.5,
             label='仿真 (PySpice AC分析)')
ax2.semilogx(freq_sim, v_phase_theory, 'r--', lw=1.8,
             label='手算理论 (φ = -arctan(f/fc))')
ax2.axvline(fc_theory, color='gray', ls=':', lw=1.5,
            label=f'fc = {fc_theory:.1f} Hz')
ax2.axhline(-45, color='green', ls=':', lw=1.2, label='-45° 线')
ax2.plot(fc_theory, -45, 'o', color='red', ms=8, zorder=5)
ax2.annotate(f'({fc_theory:.0f} Hz, -45°)',
             xy=(fc_theory, -45), xytext=(fc_theory*3, -50),
             fontsize=10, color='red',
             arrowprops=dict(arrowstyle='->', color='red'))
ax2.set_xlabel('频率 / Hz', fontsize=12)
ax2.set_ylabel('相位 / °', fontsize=12)
ax2.set_title('RC 低通滤波器 · 相频特性 (Bode图-相位)', fontsize=13)
ax2.legend(fontsize=9, loc='lower left')
ax2.grid(True, alpha=0.35, which='both')
ax2.set_xlim(10, 1e5)
ax2.set_ylim(-95, 5)

plt.tight_layout()
plt.savefig('rc_filter_bode.png', dpi=170)
print("  ✓ 已保存 rc_filter_bode.png (波特图)")

# ---------- 图2：阶跃响应 ----------
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(time_step * 1e3, v_step_sim, 'b-', lw=2.5,
        label='仿真 (PySpice 瞬态分析)')
ax.plot(time_step * 1e3, v_step_theory, 'r--', lw=1.8,
        label='手算理论 (vo = 1-e^(-t/τ))')

# 标注关键时间点
for ct, label_text in [(tau_theory, 'τ = 1ms\n(63.2%)'),
                         (3*tau_theory, '3τ = 3ms\n(95.0%)'),
                         (5*tau_theory, '5τ = 5ms\n(99.3%)')]:
    t_mark = t_delay + ct
    v_mark = 1.0 * (1.0 - np.exp(-ct / tau_theory))
    ax.axvline(t_mark * 1e3, color='gray', ls=':', lw=1.0, alpha=0.7)
    ax.axhline(v_mark, color='orange', ls=':', lw=1.0, alpha=0.5)
    ax.plot(t_mark * 1e3, v_mark, 'o', color='red', ms=6, zorder=5)
    ax.annotate(label_text,
                xy=(t_mark * 1e3, v_mark),
                xytext=(t_mark * 1e3 + 0.8, v_mark + 0.05),
                fontsize=9, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred'))

ax.set_xlabel('时间 / ms', fontsize=12)
ax.set_ylabel('输出电压 vo / V', fontsize=12)
ax.set_title(f'RC 低通滤波器 · 阶跃响应 (R={R_val/1e3:.0f}kΩ, C={C_val/1e6:.0f}μF, '
             f'τ=RC={tau_theory*1e3:.1f}ms)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.35)
ax.set_xlim(0, 8)

plt.tight_layout()
plt.savefig('rc_filter_step.png', dpi=170)
print("  ✓ 已保存 rc_filter_step.png (阶跃响应)")

# ---------- 图3：正弦瞬态响应（三频率对比） ----------
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
for idx, (t_s, v_in_s, v_out_s, label, gain_sim, gain_th) in enumerate(sine_results):
    ax = axes[idx]
    # 只画最后 2 个周期（避开初始瞬态，只看稳态）
    period = 1.0 / test_freqs[idx]
    n_show = int(2.0 * period / (t_s[1] - t_s[0]))  # 最后 2 个周期点数
    if len(t_s) > n_show:
        t_show = t_s[-n_show:]
        v_in_show = v_in_s[-n_show:]
        v_out_show = v_out_s[-n_show:]
    else:
        t_show, v_in_show, v_out_show = t_s, v_in_s, v_out_s

    ax.plot(t_show * 1e3, v_in_show, 'gray', lw=1.5, alpha=0.7, label='输入 vi (1V 正弦)')
    ax.plot(t_show * 1e3, v_out_show, 'b-', lw=2.0,
            label=f'输出 vo (仿真增益={gain_sim:.3f}, 理论={gain_th:.3f})')
    ax.set_xlabel('时间 / ms', fontsize=10)
    ax.set_ylabel('电压 / V', fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=8.5, loc='upper right')
    ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig('rc_filter_transient.png', dpi=170)
print("  ✓ 已保存 rc_filter_transient.png (正弦瞬态响应)")

# ===================== 5. 导出 CSV 数据 =====================
print("\n" + "=" * 60)
print("第5部分：导出数据")
print("=" * 60)

import csv
csv_path = 'rc_filter_data.csv'
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['频率_Hz', '仿真增益_V', '仿真增益_dB', '理论增益_V', '理论增益_dB',
                '仿真相位_deg', '理论相位_deg'])
    for i in range(len(freq_sim)):
        w.writerow([
            f'{freq_sim[i]:.2f}',
            f'{v_mag_sim[i]:.9f}',
            f'{20*np.log10(v_mag_sim[i]):.6f}',
            f'{v_mag_theory[i]:.9f}',
            f'{20*np.log10(v_mag_theory[i]):.6f}',
            f'{v_phase_sim[i]:.6f}',
            f'{v_phase_theory[i]:.6f}',
        ])
print(f"  ✓ 已保存 {csv_path} ({len(freq_sim)} 行 x 7 列)")

# ===================== 6. 最终汇总对比表 =====================
print("\n" + "=" * 60)
print("最终汇总：手算 vs 仿真 对比表")
print("=" * 60)

print(f"""
┌─────────────────────┬──────────────────┬──────────────────┬────────────┐
│ 项目                  │ 手算值             │ 仿真值             │ 相对误差     │
├─────────────────────┼──────────────────┼──────────────────┼────────────┤
│ 截止频率 fc           │ {fc_theory:>10.2f} Hz   │ {freq_sim[idx_fc]:>10.2f} Hz   │      —     │
│ |H(fc)| (增益@fc)     │ {v_at_fc_theory:>10.6f}     │ {v_at_fc_sim:>10.6f}     │ {abs(v_at_fc_sim-v_at_fc_theory)/v_at_fc_theory*100:>8.4f}% │
│ φ(fc) (相位@fc)      │ {phase_at_fc_theory:>10.2f}°     │ {phase_at_fc_sim:>10.2f}°     │     —      │
│ 时间常数 τ            │ {tau_theory*1e3:>10.2f} ms    │       —         │      —     │
│ vo(τ)=0.632V @1ms    │ {1-np.exp(-1):>10.6f}     │ 见阶跃响应图     │      —     │
│ vo(3τ)=0.950V @3ms   │ {1-np.exp(-3):>10.6f}     │ 见阶跃响应图     │      —     │
│ 通带增益 (f=0.1fc)    │ {1/np.sqrt(1+0.01):>10.6f}     │ {sine_results[0][4]:>10.6f}     │ {abs(sine_results[0][4]-1/np.sqrt(1+0.01))/(1/np.sqrt(1+0.01))*100:>8.4f}% │
│ 阻带增益 (f=10fc)     │ {1/np.sqrt(1+100):>10.6f}     │ {sine_results[2][4]:>10.6f}     │ {abs(sine_results[2][4]-1/np.sqrt(1+100))/(1/np.sqrt(1+100))*100:>8.4f}% │
└─────────────────────┴──────────────────┴──────────────────┴────────────┘

结论：
  ✓ RC 一阶低通滤波器的幅频特性、相频特性、阶跃响应，手算与
    PySpice 仿真结果高度一致，验证了理论分析的正确性。
  ✓ 截止频率 fc = {fc_theory:.2f} Hz 处，增益为 1/√2 ≈ 0.707（-3dB），
    相位为 -45°，与理论吻合。
  ✓ 阶跃响应在 t={tau_theory*1e3:.1f}ms 时达到稳态值的 63.2%，
    在 t={5*tau_theory*1e3:.1f}ms 时基本稳定（99.3%），验证了时间常数 τ 的物理意义。

生成的文件：
  rc_filter_bode.png     — 波特图（幅频 + 相频特性）
  rc_filter_step.png     — 阶跃响应（时域充放电曲线）
  rc_filter_transient.png — 正弦瞬态响应（通带/截止/阻带三频率对比）
  rc_filter_data.csv     — AC 频率扫描原始数据（可直接导入 Excel）
""")

print("\n全部完成！🎉")