import nidaqmx
import numpy as np
import time

max_voltage = 5
min_voltage = 0
# Create a task to handle the output channel
with nidaqmx.Task() as task:
    # task.ao_channels.add_ao_voltage_chan("Dev1/ao0", terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
    task.ao_channels.add_ao_voltage_chan("Dev1/ao0", max_val= max_voltage, min_val= min_voltage)
    task.ao_channels.add_ao_voltage_chan("Dev1/ao1", max_val=max_voltage, min_val=min_voltage)

    t = 0
    frequency = 0.071
    while True:
        signal = 2 * np.sin(2 * np.pi * frequency * t) + 2
        signal_2 = signal
        signals = np.asarray([signal, signal_2])
        t += 1
        task.write(signals, auto_start=True)
        time.sleep(0.05)