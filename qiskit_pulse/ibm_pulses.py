"""TODO
"""
import qiskit
import qiskit_ibm_provider
import qiskit.pulse as pulse
import numpy as np

def sequence_to_schedule(
        backend: qiskit_ibm_provider.ibm_backend.IBMBackend,
        sequence: list[qiskit.qobj.pulse_qobj.PulseQobjInstruction],
    ) -> qiskit.pulse.schedule.ScheduleBlock:
    """TODO
    """
    pulse_description = {}
    duration_offsets = {}
    with pulse.build(backend, name='custom_sched') as sched:
        for inst in sequence:
            if inst.ch not in duration_offsets:
                duration_offsets[inst.ch] = 0
            if inst.ch not in pulse_description:
                pulse_description[inst.ch] = []

            if inst.ch[0] == 'd':
                channel = pulse.DriveChannel(int(inst.ch[1]))
            else:
                channel = pulse.ControlChannel(int(inst.ch[1]))

            desired_start_time = inst.t0
            start_time_diff = desired_start_time - duration_offsets[inst.ch]
            if start_time_diff != 0:
                sched += pulse.Delay(start_time_diff, channel)
                duration_offsets[inst.ch] += start_time_diff
                pulse_description[inst.ch].append(('delay', start_time_diff))

            if inst.name == 'fc':
                sched += pulse.ShiftPhase(inst.phase, channel)
                pulse_description[inst.ch].append((inst.name, inst.phase))
            elif inst.name == 'parametric_pulse':
                if inst.pulse_shape == 'gaussian_square':
                    pulse.play(pulse.GaussianSquare(inst.parameters['duration'], inst.parameters['amp'], inst.parameters['sigma'], inst.parameters['width']), channel)
                elif inst.pulse_shape == 'drag':
                    pulse.play(pulse.Drag(inst.parameters['duration'], inst.parameters['amp'], inst.parameters['sigma'], inst.parameters['beta']), channel)
                else:
                    print('unexpected shape:')
                    print(inst)
                duration_offsets[inst.ch] += inst.parameters['duration']
                pulse_description[inst.ch].append((inst.pulse_shape, inst.parameters))
            else:
                print('unexpected pulse type:')
                print(inst)
    return sched, pulse_description

def get_default_pulse(
        backend: qiskit_ibm_provider.ibm_backend.IBMBackend, 
        pulse_name: str,
        qubits: list[int],
    ) -> tuple[list[qiskit.qobj.pulse_qobj.PulseQobjInstruction], 
               qiskit.pulse.schedule.ScheduleBlock, 
               dict[str, list[tuple[str, float]]]]:
    """TODO
    """
    # get instruction sequence
    sequence: list[qiskit.qobj.pulse_qobj.PulseQobjInstruction] | None = None
    for pulse_def in backend.defaults().cmd_def:
        if pulse_def.name == pulse_name and pulse_def.qubits == qubits:
            sequence = pulse_def.sequence
            break
    if sequence is None:
        raise Exception(f'pulse {pulse_name} not found for qubits {qubits}')
    sequence.sort(key = lambda x : x.t0)

    sched, pulse_description = sequence_to_schedule(backend, sequence)

    return sequence, sched, pulse_description