"""MLEV-4 type T2 preparation block."""

from copy import deepcopy

import numpy as np
import pypulseq as pp

from sequences.utils import sys_defaults


def add_composite_refocusing_block(
    seq: pp.Sequence,
    system: pp.Opts,
    duration_180: float,
    negative_amp: bool = False,
) -> tuple[pp.Sequence, float, float]:
    """Add a 90°x, +/-180°y, 90°x refocusing block to a sequence.

    Parameters
    ----------
    seq
        PyPulseq Sequence object.
    system
        PyPulseq system limit object. Must have rf_dead_time attribute != None.
    duration_180
        Duration of 180° refocussing block pulse (in seconds). The 90° pulses have half this duration.
    negative_amp
        Toggles negative amplitude for 180°y pulse. By default, positive amplitudes are used.

    Returns
    -------
    seq
        PyPulseq Sequence object
    block_duration
        Duration of the composite refocusing block (in seconds).
    time_to_midpoint
        Time from start of the block to the midpoint of the 180° pulse (in seconds).
        This is not necessarily the center of the block, depending on rf_dead_time and rf_ringdown_time.

    Raises
    ------
    ValueError
        If rf_dead_time is not set in system limits.
    """
    # ensure rf_dead_time is not None
    if system.rf_dead_time is None:
        raise ValueError('rf_dead_time must be provided in system limits.')

    # set rf_ringdown_time to 0 within this preparation block, since no ADC events are used
    system = deepcopy(system)
    system.rf_ringdown_time = 0

    # define flip angles and durations of RF pulses
    flip_angles = [90, 180, 90]
    durations = [duration_180 / 2, duration_180, duration_180 / 2]

    # set phases of RF pulses according to negative_amp flag
    if not negative_amp:
        phases = [0, 90, 0]
    else:
        phases = [180, 270, 180]

    # get current duration of sequence before adding composite refocusing block
    time_start = sum(seq.block_durations.values())

    # add RF pulses to sequence
    for fa, phase, dur in zip(flip_angles, phases, durations, strict=True):
        rf = pp.make_block_pulse(
            flip_angle=fa * np.pi / 180,
            delay=system.rf_dead_time,
            duration=dur,
            phase_offset=phase * np.pi / 180,
            system=system,
            use='preparation',
        )
        seq.add_block(rf)

    # calculate total block duration
    block_duration = sum(seq.block_durations.values()) - time_start

    # calculate time from start to midpoint of block
    time_to_midpoint = system.rf_dead_time + durations[0] + system.rf_dead_time + durations[1] / 2

    return (seq, block_duration, time_to_midpoint)


def add_t2prep(
    seq: pp.Sequence | None = None,
    system: pp.Opts | None = None,
    echo_time: float = 0.1,
    duration_180: float = 1e-3,
    add_spoiler: bool = True,
    spoiler_ramp_time: float = 6e-4,
    spoiler_flat_time: float = 6e-3,
) -> tuple[pp.Sequence, float]:
    """Add a MLEV-4 type T2 preparation block to a sequence.

    The MLEV-4 T2 prep block consists of a (90x, 180y, 180y, -180y, -180y, 270x, -360x) pulse pattern [Levett81]_.
    All 180°y pulses are realized using (90x, 180y, 90x) composite pulses. [Brittain95]_.

    The 'use' attribute of all RF pulses is set to "preparation" to ignore the pulses in the PyPulseq TE calculations.

    Parameters
    ----------
    seq
        PyPulseq Sequence object.
    system
        PyPulseq system limit object. Must have rf_dead_time attribute != None.
    echo_time
        Desired echo time (TE) of the block (in seconds).
        TE is defined as the time between the center of the excitation pulse and the center of the 270° tip-up pulse.
        Therefore, the total duration of the T2 prep block is always longer than the echo time.
    duration_180
        Duration of 180° refocussing pulse (in seconds).
        The duration of other pulses is scaled linearly based on their flip angles.
        For example:
            A 90° pulse will have half the duration of a 180° pulse.
            A 360° pulse will have twice the duration of a 180° pulse.
    add_spoiler
        Toggles addition of spoiler gradients at the end of the block.
        The spoiler does not effect the echo time, but increases the total duration of the T2 prep block.
    spoiler_ramp_time
        Duration of gradient spoiler ramps (in seconds).
    spoiler_flat_time
        Duration of gradient spoiler plateau (in seconds).

    Returns
    -------
    seq
        PyPulseq Sequence object.
    block_duration
        Duration of the complete T2 preparation block (in seconds).

    Raises
    ------
    ValueError
        If system limits are provided, but rf_dead_time attribute is not set.
        If desired echo_time is too short to create the T2 preparation block.

    References
    ----------
    .. [Levett81] Levitt, M. H., & Freeman, R. (1981). NMR population inversion using a composite pulse.
       Journal of Magnetic Resonance, 43(1), 65-80.

    .. [Brittain95] Brittain, J. H., Hu, B. S., Wright, G. A., Meyer, C. H., Macovski, A., & Nishimura, D. G. (1995).
       Coronary angiography with magnetization-prepared T2 contrast. Magnetic Resonance in Medicine, 33(5), 689-696.
    """
    # set system to default if not provided
    if system is None:
        system = sys_defaults

    # ensure rf_dead_time is not None
    if system.rf_dead_time is None:
        raise ValueError('rf_dead_time must be provided in system limits.')

    # set rf_ringdown_time to 0 within this preparation block, since no ADC events are used
    system = deepcopy(system)
    system.rf_ringdown_time = 0

    if seq is None:
        seq = pp.Sequence(system=system)

    # get current duration of sequence before adding T2prep block
    time_start = sum(seq.block_durations.values())

    # Create 90°x excitation pulse
    rf_90 = pp.make_block_pulse(
        flip_angle=np.pi / 2,
        delay=system.rf_dead_time,
        duration=duration_180 / 2,
        system=system,
        use='preparation',
    )

    # add 90°x pulse to sequence
    seq.add_block(rf_90)

    # Calculate delay before 1st MLEV-4 refocusing pulse
    tau1 = (
        echo_time / 8
        - duration_180 / 4  # half duration of 90° excitation pulse
        - (system.rf_dead_time + duration_180 / 2)  # duration of 90° pulse in composite refocusing block
        - (system.rf_dead_time + duration_180 / 2)  # half duration of 180° pulse in composite refocusing block
    )

    if tau1 < 0:
        raise ValueError(f'Desired echo time ({echo_time * 1000:.2f} ms) is too short to create the T2 prep block.')

    # add delay tau1 to sequence
    seq.add_block(pp.make_delay(tau1))

    # add first MLEV-4 refocusing pulse
    seq, refoc_dur, time_to_midpoint = add_composite_refocusing_block(
        system=system,
        duration_180=duration_180,
        seq=seq,
        negative_amp=False,
    )

    # add delay before 2nd MLEV-4 refocusing pulse
    tau2 = (
        echo_time / 4
        - (refoc_dur - time_to_midpoint)  # time from midpoint of 180° pulse in 1st refocussing block to end of block
        - time_to_midpoint  # time from start of 2nd refocusing block to midpoint of 180° pulse
    )

    if tau2 < 0:
        raise ValueError(f'Desired echo time ({echo_time * 1000:.2f} ms) is too short to create the T2 prep block.')

    # add delay tau2 to sequence
    seq.add_block(pp.make_delay(tau2))

    # add second MLEV-4 refocusing pulse
    seq, _, _ = add_composite_refocusing_block(
        system=system,
        duration_180=duration_180,
        seq=seq,
        negative_amp=False,
    )

    # add delay before 3rd MLEV-4 refocusing pulse. The delay time is given by tau2 as well.
    seq.add_block(pp.make_delay(tau2))

    # add third MLEV-4 refocusing pulse
    seq, _, _ = add_composite_refocusing_block(
        system=system,
        duration_180=duration_180,
        seq=seq,
        negative_amp=True,
    )

    # add delay before 4th MLEV-4 refocusing pulse. The delay time is given by tau2 as well.
    seq.add_block(pp.make_delay(tau2))

    # add fourth MLEV-4 refocusing pulse
    seq, _, _ = add_composite_refocusing_block(
        system=system,
        duration_180=duration_180,
        seq=seq,
        negative_amp=True,
    )

    # add delay before first tip-up pulse
    tau3 = (
        echo_time / 8
        - (refoc_dur - time_to_midpoint)  # time from midpoint to end of 4th refocusing block
        - (system.rf_dead_time + duration_180 / 2 * 3 / 2)  # half duration of 270° pulse)
    )

    if tau3 < 0:
        raise ValueError(f'Desired echo time ({echo_time * 1000:.2f} ms) is too short to create the T2 prep block.')

    # add delay tau3 to sequence
    seq.add_block(pp.make_delay(tau3))

    # create 270° pulse of composite tip-up pulse (270°x + [-360]°x)
    rf_tip_up_270 = pp.make_block_pulse(
        flip_angle=3 * np.pi / 2,
        delay=system.rf_dead_time,
        duration=duration_180 / 2 * 3,
        system=system,
        use='preparation',
    )

    # create -360° pulse of composite tip-up pulse (270°x + [-360]°x)
    rf_tip_up_360 = pp.make_block_pulse(
        flip_angle=-2 * np.pi,
        delay=system.rf_dead_time,
        duration=duration_180 * 2,
        system=system,
        use='preparation',
    )

    # add composite tip-up pulse to sequence
    seq.add_block(rf_tip_up_270)
    seq.add_block(rf_tip_up_360)

    # add spoiler gradient if requested
    if add_spoiler:
        gz_spoil = pp.make_trapezoid(
            channel='z',
            amplitude=0.4 * system.max_grad,
            flat_time=spoiler_flat_time,
            rise_time=spoiler_ramp_time,
            system=system,
        )
        seq.add_block(gz_spoil)

    block_duration = sum(seq.block_durations.values()) - time_start

    return (seq, block_duration)
