import nidaqmx
import numpy as np

max_voltage = 5
min_voltage = 0
# Create a task to handle the output channel
with nidaqmx.Task() as task:
    # task.ao_channels.add_ao_voltage_chan("Dev1/ao0", terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
    task.ao_channels.add_ao_voltage_chan("Dev1/ao0", max_val= max_voltage, min_val= min_voltage)
    task.ao_channels.add_ao_voltage_chan("Dev1/ao1", max_val= max_voltage, min_val= min_voltage)
    # Generate a sine wave signal
    # frequency = 1000  # Hz
    # frequency = 0.071
    frequency = 0.08
    duration = 500  # seconds
    # sample_rate = 10000  # samples/second
    sample_rate= 100
    t = np.linspace(0, duration, duration * sample_rate, endpoint=False)
    signal = 2 * np.sin(2 * np.pi * frequency * t) + 2
    signal_2 = signal
    signals = np.asarray([signal,signal_2])
    # duty_cycle = 0.5
    # signal_2 = 2.5 * (np.mod(np.floor(2 * frequency * t), 2) == 0) + 2.5 * duty_cycle

    # Write the signal to the output channel
    task.write(signals, auto_start=True)
#
# # Create a task to handle the output channels
# with nidaqmx.Task() as task:
#     task.ao_channels.add_ao_voltage_chan("Dev1/ao0", terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
#     task.ao_channels.add_ao_voltage_chan("Dev1/ao1", terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
#
#     # Generate a sine wave signal
#     frequency1 = 1000  # Hz of signal
#     duration = 1  # seconds
#     sample_rate = 10000  # samples/second for output channel (f sampling)
#     t = np.linspace(0, duration, duration * sample_rate, endpoint=False)
#     signal1 = 5 * np.sin(2 * np.pi * frequency1 * t)
#
#     # Generate a square wave signal
#     frequency2 = 500  # Hz
#     duty_cycle = 0.5
#     signal2 = 2.5 * (np.mod(np.floor(2 * frequency2 * t), 2) == 0) + 2.5 * duty_cycle
#
#     # Write the signals to both output channels simultaneously
#     task.write([signal1, signal2], auto_start=True, timeout=10.0)