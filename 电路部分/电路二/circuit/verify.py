import os
DLL_DIR = r'C:\Users\yang_\ngspice-47_dll_64\Spice64_dll\dll-vs'
os.add_dll_directory(DLL_DIR)
os.add_dll_directory(r'C:\Windows\System32')
os.environ['PATH'] = DLL_DIR + os.pathsep + os.environ.get('PATH', '')
os.environ['NGSPICE_LIBRARY_PATH'] = os.path.join(DLL_DIR, 'ngspice{}.dll')
os.environ['SPICE_LIB_DIR'] = os.path.abspath(os.path.join(DLL_DIR, '..'))

from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
NgSpiceShared.NGSPICE_PATH = os.path.abspath(os.path.join(DLL_DIR, '..'))
NgSpiceShared.LIBRARY_PATH = os.path.join(DLL_DIR, 'ngspice{}.dll')

circuit = Circuit('Task2')
circuit.V('s', 'n1', circuit.gnd, 10 @ u_V)
circuit.R('1', 'n1', 'a', 5 @ u_Ohm)
circuit.R('2', 'a', circuit.gnd, 10 @ u_Ohm)
circuit.VCCS('1', 'a', circuit.gnd, 'n1', 'a', 0.1 @ u_S)
circuit.R('L', 'a', circuit.gnd, 5 @ u_Ohm)

import numpy as np
sim = circuit.simulator()
op = sim.operating_point()
va = float(np.array(op['a']).item())
print('仿真 v(a) =', va, 'V')
print('理论值    = 2.500000 V')
