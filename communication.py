import nidaqmx
import numpy as np

# # Create a task to handle the output channel
# with nidaqmx.Task() as task:
#     task.ao_channels.add_ao_voltage_chan("Dev1/ao0", terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
#
#     # Generate a sine wave signal
#     frequency = 1000  # Hz
#     duration = 1  # seconds
#     sample_rate = 10000  # samples/second
#     t = np.linspace(0, duration, duration * sample_rate, endpoint=False)
#     signal = 5 * np.sin(2 * np.pi * frequency * t)
#
#     # Write the signal to the output channel
#     task.write(signal, auto_start=True)

# Create a task to handle the output channels
with nidaqmx.Task() as task:
    task.ao_channels.add_ao_voltage_chan("Dev1/ao0", terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
    task.ao_channels.add_ao_voltage_chan("Dev1/ao1", terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)

    # Generate a sine wave signal
    frequency1 = 1000  # Hz of signal
    duration = 1  # seconds
    sample_rate = 10000  # samples/second for output channel (f sampling)
    t = np.linspace(0, duration, duration * sample_rate, endpoint=False)
    signal1 = 5 * np.sin(2 * np.pi * frequency1 * t)

    # Generate a square wave signal
    frequency2 = 500  # Hz
    duty_cycle = 0.5
    signal2 = 2.5 * (np.mod(np.floor(2 * frequency2 * t), 2) == 0) + 2.5 * duty_cycle

    # Write the signals to both output channels simultaneously
    task.write([signal1, signal2], auto_start=True, timeout=10.0)