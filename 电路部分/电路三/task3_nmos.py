# -*- coding: utf-8 -*-
"""
============================================================
进阶挑战一 · 任务③ NMOS 共源级放大电路 —— 完整仿真验证
============================================================
功能：手算对照 + PySpice(ngspice)仿真 + 出图 + 对比表

电路（题卡）：
    VDD=5V, Rg1=60k, Rg2=40k, Rd=2k
    NMOS: K=0.8 mA/V², Vth=1V, λ=0.02/V
    vi 经耦合电容 Cb1 到栅极（栅极由 Rg1/Rg2 分压偏置到 2V）
    漏极经 Rd 接 VDD，源极接地，输出 vo 从漏极取出

运行：py -3.11 task3_nmos.py
生成：nmos_transfer.png / nmos_output.png / nmos_gain.png
      nmos_transient.png / nmos_qpoint.csv

============================================================
★ 头号坑提醒：K 与 KP 的 2 倍关系
-------------------------------------------------------------
  手算用写法 A：  Id = K · Vov²          （题给 K = 0.8 mA/V²）
  ngspice 用：    Id = ½ · KP · (W/L) · Vov² · (1 + λ·Vds)

  要让两者相等：  KP · (W/L) = 2K = 1.6 mA/V²    ← 不是 K！

  本脚本取：      KP = 160 μA/V²,  W/L = 10/1
                  → KP·(W/L) = 160μ × 10 = 1.6 mA/V²  ✅
============================================================
"""

import os
import csv

# ===================== Ngspice DLL 路径配置 =====================
# 如果你的 ngspice 解压位置不同，改这一行。
DLL_DIR = r'C:\Users\yang_\ngspice-47_dll_64\Spice64_dll\dll-vs'

# ---- 模式 A：Shared（直接加载 ngspice.dll，速度快）----
# 如果报 OSError: cannot load library ... error 0x7e，就改用下面的模式 B
if os.path.exists(DLL_DIR):
    os.add_dll_directory(DLL_DIR)                      # 让系统能找到 libomp140 等依赖
    os.add_dll_directory(r'C:\Windows\System32')
    os.environ['PATH'] = DLL_DIR + os.pathsep + os.environ.get('PATH', '')
    os.environ['NGSPICE_LIBRARY_PATH'] = os.path.join(DLL_DIR, 'ngspice{}.dll')
else:
    print("⚠ DLL 路径不存在，尝试直接加载（若失败请改用模式 B）")

# ---- 模式 B：subprocess（调 ngspice.exe，最稳）----
# 用法：把上面 模式 A 的整段注释掉，然后取消下面 3 行的注释，
#      并把路径改成你实际的 ngspice.exe 位置。
# NGSPICE_EXE = r'C:\Users\yang_\ngspice-47_dll_64\Spice64\bin\ngspice.exe'
# os.environ['NGSPICE_COMMAND'] = f'"{NGSPICE_EXE}" -b'
# USE_SUBPROCESS = True

import numpy as np
import matplotlib
matplotlib.use('Agg')                          # 非交互后端，直接存图
import matplotlib.pyplot as plt
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

# 若用模式 A，再显式指定一次 LIBRARY_PATH
if os.path.exists(DLL_DIR):
    try:
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared
        NgSpiceShared.LIBRARY_PATH = os.path.join(DLL_DIR, 'ngspice{}.dll')
    except Exception:
        pass

# ===================== 中文字体 =====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False     # 负号显示成方块的问题

# ===================== 题卡参数 =====================
VDD   = 5.0          # 电源电压 5 V
RG1   = 60e3         # 栅极上偏电阻 60 kΩ
RG2   = 40e3         # 栅极下偏电阻 40 kΩ
RD    = 2e3          # 漏极电阻 2 kΩ
K     = 0.8e-3       # 跨导参数 K = 0.8 mA/V²（写法 A 约定：Id = K·Vov²）
VTH   = 1.0          # 阈值电压 1 V
LAMBDA = 0.02        # 沟道长度调制系数 0.02 /V
CB1   = 10e-6        # 输入耦合电容 10 μF（题卡未给，自选，理由见手册第6章坑7）

# ngspice 模型参数：KP·(W/L) 必须 = 2K
W_VAL = 10e-6        # 沟道宽度 10 μm
L_VAL = 1e-6         # 沟道长度 1 μm
KP_VAL = 160e-6      # KP = 160 μA/V²  → KP·(W/L) = 160μ × 10 = 1.6 mA/V² = 2K ✅

# ===================== 手算值（第2章推导结果） =====================
VG_CALC   = VDD * RG2 / (RG1 + RG2)            # 栅极分压 = 2.000 V
VOV_CALC  = VG_CALC - VTH                       # 过驱动电压 = 1.000 V
ID_CALC   = K * VOV_CALC**2                     # Id = K·Vov² = 0.800 mA（忽略λ）
VD_CALC   = VDD - ID_CALC * RD                  # Vd = 5 − 0.8m×2k = 3.400 V
GM_CALC   = 2 * K * VOV_CALC                    # gm = 2K·Vov = 1.600 mS
RO_CALC   = 1.0 / (LAMBDA * ID_CALC)            # ro = 1/(λ·Id) = 62.5 kΩ
ROUT_CALC = (RD * RO_CALC) / (RD + RO_CALC)     # Rd ∥ ro = 1.938 kΩ
AV_CALC   = -GM_CALC * ROUT_CALC                # Av = −3.10

# 含 λ 修正的精确值（仿真会对上这一组）
ID_LAM = ID_CALC * (1 + LAMBDA * VD_CALC)       # 0.8544 mA
VD_LAM = VDD - ID_LAM * RD                      # 3.291 V
RO_LAM = 1.0 / (LAMBDA * ID_LAM)                # 58.5 kΩ
AV_LAM = -GM_CALC * (RD * RO_LAM) / (RD + RO_LAM)   # −3.09

print("=" * 66)
print("任务③ NMOS 共源级放大电路 · 手算 vs 仿真")
print("=" * 66)
print(f"\n【ngspice 模型设置】KP={KP_VAL*1e6:.0f} μA/V², W/L={W_VAL/L_VAL:.0f}"
      f"  → KP·(W/L)={KP_VAL*W_VAL/L_VAL*1e3:.2f} mA/V² (=2K) ✅")
print(f"\n【手算 · 简化（忽略λ对Id）】")
print(f"  Vg  = {VG_CALC:.4f} V")
print(f"  Vov = {VOV_CALC:.4f} V")
print(f"  Id  = {ID_CALC*1e3:.4f} mA")
print(f"  Vd  = {VD_CALC:.4f} V")
print(f"  gm  = {GM_CALC*1e3:.4f} mS")
print(f"  ro  = {RO_CALC/1e3:.2f} kΩ")
print(f"  Av  = {AV_CALC:.4f}")
print(f"\n【手算 · 含λ修正（仿真应接近这一组）】")
print(f"  Id  = {ID_LAM*1e3:.4f} mA")
print(f"  Vd  = {VD_LAM:.4f} V")
print(f"  ro  = {RO_LAM/1e3:.2f} kΩ")
print(f"  Av  = {AV_LAM:.4f}")


def build_amp():
    """
    构建共源级放大电路（主电路，含偏置）。

    拓扑：
        VDD ── Rg1 ──┬── Rg2 ── gnd          ← 栅极分压，垫到 2V
                     │
        vi ── Cb1 ───┴── 栅极 g
                                              ← Cb1 隔直通交
        VDD ── Rd ──┬── 漏极 d ── vo（输出）
                    │
                   M1 (NMOS)
                    │
                  源极 s ── gnd
    """
    c = Circuit('NMOS_CS_Amp')
    c.V('dd', 'vdd', c.gnd, VDD @ u_V)                  # 电源 VDD = 5V
    c.R('g1', 'vdd', 'vgate', RG1 @ u_Ohm)              # Rg1 = 60k（上偏）
    c.R('g2', 'vgate', c.gnd, RG2 @ u_Ohm)              # Rg2 = 40k（下偏）
    c.R('d', 'vdd', 'vdrain', RD @ u_Ohm)               # Rd = 2k（漏极负载）
    # 输入信号源 + 耦合电容：DC 被 Cb1 隔离，栅极 DC 只由 Rg1/Rg2 决定
    c.SinusoidalVoltageSource('in', 'vin', c.gnd,
                              dc_offset=0 @ u_V,
                              amplitude=1e-3 @ u_V,     # 1 mV 小信号
                              frequency=10e3 @ u_Hz)    # 10 kHz
    c.C('b1', 'vin', 'vgate', CB1 @ u_F)                # Cb1 = 10 μF
    # NMOS 模型：LEVEL=1(平方律) VTO=Vth, KP, LAMBDA=λ
    # 关键：KP·(W/L) 必须 = 2K = 1.6 mA/V²
    c.model('mynmos', 'NMOS', LEVEL=1, VTO=VTH,
            KP=KP_VAL, LAMBDA=LAMBDA)
    # M 器件：端子顺序 = 漏(d) 栅(g) 源(s) 体(b)
    #         源极接地，NMOS 体端也接地
    c.MOSFET('1', 'vdrain', 'vgate', c.gnd, c.gnd,
             model='mynmos', w=W_VAL, l=L_VAL)
    return c


# ===================== 1. 直流工作点（Q 点） =====================
print("\n" + "=" * 66)
print("第1部分：直流工作点 .op")
print("=" * 66)

circuit = build_amp()
sim = circuit.simulator(temperature=25, nominal_temperature=25)
op = sim.operating_point()

Vg_sim = float(np.array(op['vgate']).item())       # 栅极直流电压
Vd_sim = float(np.array(op['vdrain']).item())       # 漏极直流电压
Id_sim = (VDD - Vd_sim) / RD         # 漏极电流：从 Rd 压降反推（最稳）

print(f"  仿真 Vg = {Vg_sim:.4f} V     手算 {VG_CALC:.4f} V")
print(f"  仿真 Vd = {Vd_sim:.4f} V     手算 {VD_LAM:.4f} V (含λ) / {VD_CALC:.4f} V (简化)")
print(f"  仿真 Id = {Id_sim*1e3:.4f} mA    手算 {ID_LAM*1e3:.4f} mA (含λ) / {ID_CALC*1e3:.4f} mA (简化)")

# ---- 用数值微分求 gm（在 Q 点附近对栅压做 ±1 mV 扰动，测 Id 变化斜率）----
# 方法：对 Vgs 加微小扰动，通过 Rd 压降反推 Id，求斜率
def id_at_vgs(vgs):
    """给定栅压 vgs，返回漏极电流 Id（含 Rd，从压降反推）"""
    t = Circuit('Test_gm')
    t.V('dd', 'vdd', t.gnd, VDD @ u_V)
    t.R('d', 'vdd', 'vdrain', RD @ u_Ohm)
    t.V('gs', 'vgate', t.gnd, vgs @ u_V)      # 直接给定 Vgs
    t.model('mynmos', 'NMOS', LEVEL=1, VTO=VTH, KP=KP_VAL, LAMBDA=LAMBDA)
    t.MOSFET('1', 'vdrain', 'vgate', t.gnd, t.gnd,
             model='mynmos', w=W_VAL, l=L_VAL)
    s = t.simulator(temperature=25, nominal_temperature=25)
    o = s.operating_point()
    vd = float(np.array(o['vdrain']).item())
    return (VDD - vd) / RD                     # Id = 压降/Rd

dv = 1e-3                                     # 1 mV 扰动
gm_sim = (id_at_vgs(VG_CALC + dv) - id_at_vgs(VG_CALC - dv)) / (2 * dv)
print(f"  仿真 gm = {gm_sim*1e3:.4f} mS   手算 {GM_CALC*1e3:.4f} mS   "
      f"误差 {abs(gm_sim-GM_CALC)/GM_CALC*100:.2f}%")

# ---- ro：解析计算 ro = 1/(λ·Id_sim)，比数值微分更稳 ----
ro_sim = 1.0 / (LAMBDA * Id_sim)
print(f"  仿真 ro = {ro_sim/1e3:.2f} kΩ   手算 {RO_LAM/1e3:.2f} kΩ (含λ) / {RO_CALC/1e3:.2f} kΩ (简化)")

# ---- 验证饱和区 ----
Vds_sim = Vd_sim
Vov_sim = VG_CALC - VTH
sat_ok = Vds_sim >= Vov_sim
print(f"\n  【工作区验证】Vds = {Vds_sim:.4f} V, Vov = {Vov_sim:.4f} V")
print(f"    判据 Vds ≥ Vov → {Vds_sim:.4f} ≥ {Vov_sim:.4f} → "
      f"{'✅ 饱和区（放大区）' if sat_ok else '❌ 线性区！增益公式不成立'}")

# ---- 增益（由 gm、ro 合成）----
Rout_sim = (RD * ro_sim) / (RD + ro_sim)
Av_sim = -gm_sim * Rout_sim
print(f"\n  仿真 Av = {Av_sim:.4f}   手算 {AV_LAM:.4f} (含λ) / {AV_CALC:.4f} (简化)")
print(f"    误差（对含λ手算）= {abs(Av_sim-AV_LAM)/abs(AV_LAM)*100:.2f}%")

# ===================== 2. 转移特性 Id–Vgs =====================
print("\n" + "=" * 66)
print("第2部分：转移特性（扫栅压 Vgs，看 Id 怎么随阀门开度变）")
print("=" * 66)

vgs_arr = np.linspace(0, 3, 200)
id_arr = np.array([id_at_vgs(v) * 1e3 for v in vgs_arr])   # 转成 mA

plt.figure(figsize=(8.5, 5.5))
plt.plot(vgs_arr, id_arr, 'b-', lw=2.5, label='仿真 Id–Vgs')
# 标出阈值点和 Q 点
plt.axvline(VTH, color='gray', ls=':', lw=1.5, label=f'Vth = {VTH} V')
plt.plot([VG_CALC], [Id_sim * 1e3], 'o', color='red', ms=10, zorder=5)
plt.annotate(f'Q 点\n(Vgs={VG_CALC:.2f}V, Id={Id_sim*1e3:.3f}mA)',
             xy=(VG_CALC, Id_sim * 1e3),
             xytext=(VG_CALC + 0.35, Id_sim * 1e3 * 0.45),
             fontsize=10.5, color='red',
             arrowprops=dict(arrowstyle='->', color='red'))
plt.xlabel('栅源电压 Vgs / V', fontsize=12)
plt.ylabel('漏极电流 Id / mA', fontsize=12)
plt.title('NMOS 转移特性：Id 随 Vgs 变化（Q 点=2V, 饱和区）', fontsize=13)
plt.legend(fontsize=9.5)
plt.grid(alpha=0.35)
plt.xlim(0, 3)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('nmos_transfer.png', dpi=170)
print("  ✓ 已保存 nmos_transfer.png")

# ===================== 3. 输出特性 Id–Vds =====================
print("\n第3部分：输出特性（扫漏压 Vds，看饱和区与 λ 上翘）")

plt.figure(figsize=(8.5, 5.5))
for vgs_test, col in [(1.5, 'green'), (2.0, 'blue'), (2.5, 'purple')]:
    vds_arr = np.linspace(0, 5, 150)
    ids = []
    for vds in vds_arr:
        t = Circuit('Out')
        t.V('ds', 'vdrain', t.gnd, vds @ u_V)
        t.V('gs', 'vgate', t.gnd, vgs_test @ u_V)
        t.model('mynmos', 'NMOS', LEVEL=1, VTO=VTH, KP=KP_VAL, LAMBDA=LAMBDA)
        t.MOSFET('1', 'vdrain', 'vgate', t.gnd, t.gnd,
                 model='mynmos', w=W_VAL, l=L_VAL)
        s = t.simulator(temperature=25, nominal_temperature=25)
        o = s.operating_point()
        ids.append(float(np.array(-o['vdrain']).item()) * 1e3)
    plt.plot(vds_arr, ids, lw=2.2, color=col, label=f'Vgs = {vgs_test} V')
    # 标出饱和边界 Vds = Vov
    vov_t = vgs_test - VTH
    if vov_t > 0:
        plt.axvline(vov_t, color=col, ls='--', lw=1.0, alpha=0.5)

plt.axvline(VG_CALC - VTH, color='red', ls=':', lw=1.8,
            label=f'Q点饱和边界 Vov = {Vov_sim:.1f} V')
plt.plot([Vd_sim], [Id_sim * 1e3], 'o', color='red', ms=10, zorder=5)
plt.annotate(f'Q 点\n(Vds={Vd_sim:.2f}V, Id={Id_sim*1e3:.3f}mA)',
             xy=(Vd_sim, Id_sim * 1e3),
             xytext=(Vd_sim - 2.2, Id_sim * 1e3 + 0.35),
             fontsize=10.5, color='red',
             arrowprops=dict(arrowstyle='->', color='red'))
plt.xlabel('漏源电压 Vds / V', fontsize=12)
plt.ylabel('漏极电流 Id / mA', fontsize=12)
plt.title('NMOS 输出特性：饱和区曲线上翘 = λ（沟道长度调制）', fontsize=13)
plt.legend(fontsize=9.5)
plt.grid(alpha=0.35)
plt.xlim(0, 5)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('nmos_output.png', dpi=170)
print("  ✓ 已保存 nmos_output.png")

# ===================== 4. AC 分析：中频增益 =====================
print("\n" + "=" * 66)
print("第4部分：AC 扫频（找中频增益）")
print("=" * 66)

sim_ac = build_amp().simulator(temperature=25, nominal_temperature=25)
ac = sim_ac.ac(start_frequency=0.1 @ u_Hz,
               stop_frequency=10e6 @ u_Hz,
               number_of_points=200,
               variation='dec')
freq = np.array(ac.frequency)
gain = np.array(ac['vdrain']) / np.array(ac['vin'])     # 增益 = vo/vi
gain_db = 20 * np.log10(np.abs(gain))

# 读中频增益（10 kHz 附近，避开低频 Cb1 影响）
idx_mid = np.argmin(np.abs(freq - 10e3))
gain_mid = gain[idx_mid]
print(f"  中频(10kHz)增益 = {gain_mid:.4f}  ({gain_db[idx_mid]:.2f} dB)")
print(f"  手算 Av = {AV_LAM:.4f} (含λ) / {AV_CALC:.4f} (简化)")
print(f"  负号 = 反相（共源级特征）✅")

plt.figure(figsize=(8.5, 5.5))
plt.semilogx(freq, gain_db, 'b-', lw=2.5, label='仿真增益')
plt.axhline(20 * np.log10(abs(AV_LAM)), color='r', ls='--', lw=1.8,
            label=f'手算 Av={AV_LAM:.2f} ({20*np.log10(abs(AV_LAM)):.2f} dB)')
plt.axvline(10e3, color='gray', ls=':', lw=1.5, label='10 kHz（读增益处）')
plt.xlabel('频率 / Hz', fontsize=12)
plt.ylabel('增益 / dB', fontsize=12)
plt.title('共源级放大器 · 幅频响应（中频增益 ≈ −3.1，即约 9.8 dB）', fontsize=13)
plt.legend(fontsize=9.5)
plt.grid(alpha=0.35, which='both')
plt.tight_layout()
plt.savefig('nmos_gain.png', dpi=170)
print("  ✓ 已保存 nmos_gain.png")

# ===================== 5. 瞬态：看放大 + 反相 =====================
print("\n第5部分：瞬态波形（直观看到放大和反相）")

# 用较大的输入(10mV)让波形看得清，同时保证仍在小信号范围
c_tran = Circuit('NMOS_CS_Tran')
c_tran.V('dd', 'vdd', c_tran.gnd, VDD @ u_V)
c_tran.R('g1', 'vdd', 'vgate', RG1 @ u_Ohm)
c_tran.R('g2', 'vgate', c_tran.gnd, RG2 @ u_Ohm)
c_tran.R('d', 'vdd', 'vdrain', RD @ u_Ohm)
c_tran.SinusoidalVoltageSource('in', 'vin', c_tran.gnd,
                               dc_offset=0 @ u_V,
                               amplitude=10e-3 @ u_V,   # 10 mV
                               frequency=10e3 @ u_Hz)   # 10 kHz
c_tran.C('b1', 'vin', 'vgate', CB1 @ u_F)
c_tran.model('mynmos', 'NMOS', LEVEL=1, VTO=VTH, KP=KP_VAL, LAMBDA=LAMBDA)
c_tran.MOSFET('1', 'vdrain', 'vgate', c_tran.gnd, c_tran.gnd,
              model='mynmos', w=W_VAL, l=L_VAL)

sim_tr = c_tran.simulator(temperature=25, nominal_temperature=25)
n_cycles = 3
period = 1 / 10e3
tran = sim_tr.transient(step_time=(period / 500) @ u_s,
                        end_time=(n_cycles * period) @ u_s)
t_arr = np.array(tran.time)
vin_arr = np.array(tran['vin'])
vo_arr = np.array(tran['vdrain'])
# 只看稳态（后 2/3 段），去掉起始暂态
n0 = int(len(t_arr) * 0.33)
vo_ac = vo_arr[n0:] - np.mean(vo_arr[n0:])       # 去掉直流成分，只看交流摆动
vin_ac = vin_arr[n0:]
amp_out = (np.max(vo_ac) - np.min(vo_ac)) / 2
amp_in = (np.max(vin_ac) - np.min(vin_ac)) / 2
gain_tran = amp_out / amp_in
print(f"  输入幅度 = {amp_in*1e3:.3f} mV, 输出幅度 = {amp_out*1e3:.3f} mV")
print(f"  瞬态增益 = {gain_tran:.4f}（负号在波形上体现为反相）")

plt.figure(figsize=(9, 5.5))
plt.plot(t_arr * 1e3, vin_arr * 1e3, 'b-', lw=2, label='输入 vi (10 mV)')
plt.plot(t_arr * 1e3, (vo_arr - np.mean(vo_arr[n0:])) * 1e3, 'r-', lw=2,
         label=f'输出 vo（放大约 {abs(gain_tran):.1f} 倍，反相）')
plt.axhline(0, color='k', lw=0.8, alpha=0.4)
plt.xlabel('时间 / ms', fontsize=12)
plt.ylabel('电压 / mV', fontsize=12)
plt.title('共源级放大：输出被放大且相位相反（反相 = 共源级标志）', fontsize=13)
plt.legend(fontsize=10)
plt.grid(alpha=0.35)
plt.tight_layout()
plt.savefig('nmos_transient.png', dpi=170)
print("  ✓ 已保存 nmos_transient.png")

# ===================== 6. 对比表 + CSV =====================
print("\n" + "=" * 66)
print("对比表：手算 vs 仿真")
print("=" * 66)
rows = [
    ('栅极电压 Vg (V)',      VG_CALC,            Vg_sim),
    ('过驱动电压 Vov (V)',   VOV_CALC,           Vov_sim),
    ('漏极电流 Id (mA)',     ID_LAM * 1e3,      Id_sim * 1e3),
    ('漏极电压 Vd (V)',      VD_LAM,             Vd_sim),
    ('跨导 gm (mS)',         GM_CALC * 1e3,     gm_sim * 1e3),
    ('输出电阻 ro (kΩ)',     RO_LAM / 1e3,      ro_sim / 1e3),
    ('电压增益 Av',          AV_LAM,             Av_sim.real if hasattr(Av_sim, 'real') else Av_sim),
]
print(f"{'项目':<22}{'手算(含λ)':>14}{'仿真':>14}{'误差':>12}")
for name, calc, simv in rows:
    err = abs(calc - simv) / abs(calc) * 100 if calc != 0 else 0
    print(f"{name:<22}{calc:>14.4f}{simv:>14.4f}{err:>11.2f}%")

with open('nmos_qpoint.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['项目', '手算(含λ)', '手算(简化)', '仿真', '误差%'])
    w.writerow(['Vg(V)', f'{VG_CALC:.6f}', f'{VG_CALC:.6f}', f'{Vg_sim:.6f}',
                f'{abs(VG_CALC-Vg_sim)/VG_CALC*100:.3f}'])
    w.writerow(['Id(mA)', f'{ID_LAM*1e3:.6f}', f'{ID_CALC*1e3:.6f}', f'{Id_sim*1e3:.6f}',
                f'{abs(ID_LAM-Id_sim)/ID_LAM*100:.3f}'])
    w.writerow(['Vd(V)', f'{VD_LAM:.6f}', f'{VD_CALC:.6f}', f'{Vd_sim:.6f}',
                f'{abs(VD_LAM-Vd_sim)/VD_LAM*100:.3f}'])
    w.writerow(['gm(mS)', f'{GM_CALC*1e3:.6f}', f'{GM_CALC*1e3:.6f}', f'{gm_sim*1e3:.6f}',
                f'{abs(GM_CALC-gm_sim)/GM_CALC*100:.3f}'])
    w.writerow(['ro(kΩ)', f'{RO_LAM/1e3:.6f}', f'{RO_CALC/1e3:.6f}', f'{ro_sim/1e3:.6f}',
                f'{abs(RO_LAM-ro_sim)/RO_LAM*100:.3f}'])
    w.writerow(['Av', f'{AV_LAM:.6f}', f'{AV_CALC:.6f}', f'{Av_sim:.6f}',
                f'{abs(AV_LAM-Av_sim)/abs(AV_LAM)*100:.3f}'])
print("\n  ✓ 已保存 nmos_qpoint.csv")

print("\n" + "=" * 66)
print("全部完成！生成文件：")
print("  nmos_transfer.png   转移特性（Id–Vgs，标 Q 点）")
print("  nmos_output.png     输出特性（Id–Vds，看 λ 上翘）")
print("  nmos_gain.png       幅频响应（中频增益）")
print("  nmos_transient.png  瞬态波形（放大 + 反相）")
print("  nmos_qpoint.csv     Q 点对比数据")
print("=" * 66)