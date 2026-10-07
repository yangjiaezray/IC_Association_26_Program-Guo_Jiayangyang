import os
DLL_DIR = r'C:\Users\yang_\ngspice-47_dll_64\Spice64_dll\dll-vs'
os.add_dll_directory(DLL_DIR)
os.add_dll_directory(r'C:\Windows\System32')
os.environ['PATH'] = DLL_DIR + os.pathsep + os.environ.get('PATH', '')
os.environ['SPICE_LIB_DIR'] = os.path.abspath(os.path.join(DLL_DIR, '..'))
os.environ['NGSPICE_LIBRARY_PATH'] = os.path.join(DLL_DIR, 'ngspice{}.dll')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
NgSpiceShared.NGSPICE_PATH = os.path.abspath(os.path.join(DLL_DIR, '..'))
NgSpiceShared.LIBRARY_PATH = os.path.join(DLL_DIR, 'ngspice{}.dll')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

US, R1, R2, GM = 10.0, 5.0, 10.0, 0.1
UOC, REQ = 5.0, 5.0

def build_original(RL):
    c = Circuit('Orig')
    c.V('s', 'n1', c.gnd, US @ u_V)
    c.R('1', 'n1', 'a', R1 @ u_Ohm)
    c.R('2', 'a', c.gnd, R2 @ u_Ohm)
    c.VCCS('1', 'a', c.gnd, 'n1', 'a', GM @ u_S)
    c.R('L', 'a', c.gnd, RL @ u_Ohm)
    return c

def build_thevenin(RL):
    c = Circuit('Thev')
    c.V('oc', 'x', c.gnd, UOC @ u_V)
    c.R('eq', 'x', 'a', REQ @ u_Ohm)
    c.R('L', 'a', c.gnd, RL @ u_Ohm)
    return c

def run(c):
    return c.simulator(temperature=25, nominal_temperature=25).operating_point()

print("=== 求 Uoc / Isc / Req ===")
uoc = float(np.array(run(build_original(1e12))['a']).item())
isc = float(np.array(run(build_original(1e-9))['a']).item()) / 1e-9
print(f"Uoc = {uoc:.6f} V  (手算 5.000)")
print(f"Isc = {isc:.6f} A  (手算 1.000)")
print(f"Req = {uoc/isc:.6f} Ω  (手算 5.000)")

print("\n=== 扫描负载 RL ===")
RL_list = np.logspace(-1, 4, 121)
rows = []
for RL in RL_list:
    v1 = float(np.array(run(build_original(RL))['a']).item())
    v2 = float(np.array(run(build_thevenin(RL))['a']).item())
    err = abs(v1-v2)/abs(v2) if v2 else 0.0
    rows.append((RL, v1, v1/RL, v2, v2/RL, err))

print(f"{'RL/Ω':>10} {'U_原/V':>14} {'U_等效/V':>14} {'误差':>10}")
for r in rows[::20]:
    print(f"{r[0]:>10.3f} {r[1]:>14.9f} {r[3]:>14.9f} {r[5]:>10.2e}")
print(f"\n最大相对误差 = {max(r[5] for r in rows):.3e}")

RL = np.array([r[0] for r in rows])
Uo = np.array([r[1] for r in rows]); Io = np.array([r[2] for r in rows])
Ut = np.array([r[3] for r in rows]); It = np.array([r[4] for r in rows])
Po, Pt = Uo*Io, Ut*It
Pmax = UOC**2/(4*REQ)

# 图1: V-I 曲线
plt.figure(figsize=(8.5,5.8))
plt.plot(Io, Uo, 'b-', lw=3, label='原网络（含 VCCS）')
plt.plot(It, Ut, 'r--', lw=2, label=f'戴维南等效 ({UOC}V + {REQ}Ω)')
plt.xlabel('端口电流 I / A'); plt.ylabel('端口电压 U / V')
plt.title('端口 V-I 特性对比：两条曲线完全重合 → 戴维南定理成立')
plt.legend(); plt.grid(alpha=0.35)
plt.plot([0],[UOC],'o',color='gray',ms=8)
plt.annotate(f'开路点 (0, {UOC}V)', xy=(0,UOC), xytext=(0.12,UOC*0.82),
             fontsize=9.5, arrowprops=dict(arrowstyle='->'))
plt.tight_layout(); plt.savefig('thevenin_VI.png', dpi=170)
print("\n已保存 thevenin_VI.png")

# 图2: 功率曲线
plt.figure(figsize=(8.5,5.5))
plt.semilogx(RL, Po, 'b-', lw=2.6, label='负载功率（原网络）')
plt.semilogx(RL, Pt, 'r--', lw=2, label='负载功率（戴维南等效）')
plt.axvline(REQ, color='k', ls=':', label=f'RL = Req = {REQ} Ω')
plt.plot([REQ],[Pmax],'o',color='red',ms=10)
plt.annotate(f'最大功率 {Pmax:.3f} W\n@ RL = Req = {REQ} Ω',
             xy=(REQ,Pmax), xytext=(REQ*2.6,Pmax*0.72),
             fontsize=10.5, color='red', arrowprops=dict(arrowstyle='->',color='red'))
plt.xlabel('负载 RL / Ω（对数）'); plt.ylabel('负载功率 P / W')
plt.title('最大功率传输定理验证')
plt.legend(); plt.grid(alpha=0.35, which='both')
plt.tight_layout(); plt.savefig('thevenin_power.png', dpi=170)
print("已保存 thevenin_power.png")

# 图3: 电压随负载变化
plt.figure(figsize=(8.5,5.2))
plt.semilogx(RL, Uo, 'b-', lw=2.6, label='原网络')
plt.semilogx(RL, Ut, 'ro', ms=3.5, label='戴维南等效')
plt.axhline(UOC, color='gray', ls=':', label=f'Uoc = {UOC} V（RL→∞）')
plt.xlabel('负载 RL / Ω（对数）'); plt.ylabel('端口电压 U / V')
plt.title('负载变化时端口电压：原网络 vs 戴维南等效')
plt.legend(); plt.grid(alpha=0.35, which='both')
plt.tight_layout(); plt.savefig('thevenin_U_vs_RL.png', dpi=170)
print("已保存 thevenin_U_vs_RL.png")

# 导出 CSV
import csv
with open('thevenin_data.csv','w',newline='',encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['RL/Ohm','U_orig/V','I_orig/A','U_thev/V','I_thev/A','rel_err'])
    for r in rows:
        w.writerow([f'{r[0]:.6g}',f'{r[1]:.9f}',f'{r[2]:.9f}',
                    f'{r[3]:.9f}',f'{r[4]:.9f}',f'{r[5]:.3e}'])
print("已保存 thevenin_data.csv")
print("\n全部完成！")