# Baseed on Gidney--Ekera "fill-in-table.py"
# https://arxiv.org/abs/1905.09749

import math
import numpy as np


def estimate_abstract_to_physical(tof_count,
                                  abstract_qubits,
                                  measurement_depth,
                                  prefers_parallel = False,
                                  prefers_serial = False):
    for code_distance in range(25, 501):
        logical_qubit_area = code_distance**2 * 2
        ccz_factory_duration = 5*code_distance
        ccz_factory_footprint = 15*8*logical_qubit_area
        ccz_factory_volume = ccz_factory_footprint * ccz_factory_duration

        reaction_time = 10
        assert not (prefers_parallel and prefers_serial)

        if prefers_parallel:
            # Use time optimal computation.
            runtime = reaction_time * measurement_depth
            routing_overhead_factor = 1.5
        elif prefers_serial:
            # Use serial distillation.
            runtime = tof_count * ccz_factory_duration
            routing_overhead_factor = 1.01
        else:
            # Do something intermediate.
            runtime = ccz_factory_duration * measurement_depth
            routing_overhead_factor = 1.25

        data_footprint = abstract_qubits * logical_qubit_area
        data_volume = data_footprint * runtime
        unit_cells = abstract_qubits * runtime
        error_per_unit_cell = 10**-math.ceil(code_distance / 2 + 1)
        data_error = unit_cells * error_per_unit_cell
        if data_error > 0.25:
            continue

        distill_volume = tof_count * ccz_factory_volume
        factory_count = int(math.ceil(distill_volume / runtime / ccz_factory_footprint))
        distill_footprint = factory_count * ccz_factory_footprint

        total_volume = data_volume * routing_overhead_factor + distill_volume
        total_footprint = (data_footprint + distill_footprint) * routing_overhead_factor

        microseconds_per_day = 10**6 * 60 * 60 * 24
        qubit_microseconds_per_megaqubit_day = 10**6 * microseconds_per_day

        return (
            code_distance,
            factory_count,
            tof_count,
            runtime / microseconds_per_day,
            total_footprint / 10.0**6,
            total_volume / qubit_microseconds_per_megaqubit_day,
        )

    raise NotImplementedError()
