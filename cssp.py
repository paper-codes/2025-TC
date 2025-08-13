from math import comb

import numpy as np
from qat.lang.AQASM import classarith
from qat.lang.AQASM.gates import H, X, Z
from qat.lang.AQASM.program import Program
from qat.lang.AQASM.qftarith import QFT
from qat.lang.AQASM.routines import QRoutine
from qat.pylinalg import PyLinalg

from qatext.qatmgmt.program import ProgramWrapper
from qatext.qatmgmt.routines import QRoutineWrapper
from qatext.qatmgmt.sample import extract_qarray_values_by_named_qarrays
from qatext.qroutines.arith import cuccaro_arith
from qatext.qroutines.datastructure.array import contains
from qatext.qroutines.datastructure.sliding_sort_array import \
    insert as insert_ld  # ld stands for low-depth
from qatext.qroutines.datastructure.sliding_sort_array import insert_lw
from qatext.qroutines.hamming_weight_generate.bartschiE19 import generate
from qatext.qroutines.qregs_mgmt import qregs_init as qi
from qatext.qroutines.qregs_mgmt import qregs_init_bix as bix

QPU = PyLinalg()


def simulate_program(
    prw: ProgramWrapper,  # Program or Circuit
    qubits=None,
):
    cr = prw.to_circ(link=[classarith, cuccaro_arith])
    print(cr.statistics())
    job = cr.to_job(qubits=qubits)
    res = QPU.submit(job)
    for sample in res:
        result = extract_qarray_values_by_named_qarrays(
            prw._name_to_qarray, sample)
        print(sample.amplitude, result)


def update(n, k, m, insert, has_duplicates):
    qrw = QRoutineWrapper(QRoutine())

    node_s_ones = qrw.qarray_wires(k, m, "s_1", int)
    node_s_zeros = qrw.qarray_wires(n - k, m, "s_0", int)
    node_t_ones = qrw.qarray_wires(k, m, "t_1", int)
    node_t_zeros = qrw.qarray_wires(n - k, m, "t_0", int)
    alpha_ones = qrw.qarray_wires(1, m, "a_1", int)
    alpha_zeros = qrw.qarray_wires(1, m, "a_0", int)
    qrw.set_ancillae(alpha_ones)
    qrw.set_ancillae(alpha_zeros)
    wstate_ones = qrw.qarray_wires(k, 1, "w_1", str)
    wstate_zeros = qrw.qarray_wires(n - k, 1, "w_0", str)

    qrout_insert_ones = insert(k, m)
    qrout_insert_zeros = insert(n - k, m)
    qrout_contains_ones = contains(k, m, has_duplicates)
    qrout_contains_zeros = contains(n - k, m, has_duplicates)

    # copy s to t
    qrw.apply(qi.copy_array_of_registers(k, m), node_s_ones, node_t_ones)
    qrw.apply(qi.copy_array_of_registers(n - k, m), node_s_zeros, node_t_zeros)

    # generate 2 w states, one w/ k, and another one w/ n-k elements
    qrw.apply(generate(k, 1), wstate_ones)
    qrw.apply(generate(n - k, 1), wstate_zeros)
    # copy node_s_ones[j] to alpha_ones if w1[j] is 1
    for j in range(k):
        qrw.apply(
            qi.copy_register(m).ctrl(), wstate_ones[j], node_s_ones[j],
            alpha_ones)
    # delete the selected elements (in alpha_ones) from node_t_ones
    qrw.apply(qrout_insert_ones.dag(), alpha_ones, node_t_ones)
    # copy node_s_zeros[j] to alpha_zeros if w2[j] is 1
    for j in range(n - k):
        qrw.apply(
            qi.copy_register(m).ctrl(), wstate_zeros[j], node_t_zeros[j],
            alpha_zeros)
    # delete the selected elements (in alpha_zeros) from node_t_zeros
    qrw.apply(qrout_insert_zeros.dag(), alpha_zeros, node_t_zeros)

    # insert in node_s_ones the value stored in alpha_zeros, and viceversa
    qrw.apply(qrout_insert_ones, alpha_zeros, node_t_ones)
    qrw.apply(qrout_insert_zeros, alpha_ones, node_t_zeros)

    # reset ancilla
    for j in range(k):
        qrw.apply(
            qi.copy_register(m).ctrl(), wstate_ones[j], node_s_ones[j],
            alpha_ones)
    for j in range(n - k):
        qrw.apply(
            qi.copy_register(m).ctrl(), wstate_zeros[j], node_t_zeros[j],
            alpha_zeros)
    # qrw.free_ancillae(alpha_zeros)
    # qrw.free_ancillae(alpha_ones)
    # qbit_out = qrw.get_free_ancillae(1)
    # the previous does not seem to work, doing it manually
    qbit_out = alpha_zeros[0][0]

    # reset wstates
    for j in range(k):
        # check if node_s_ones[j] is present in node_t_ones and, if not, apply
        # X to w[j]
        qrw.apply(qrout_contains_ones, node_s_ones[j], node_t_ones, qbit_out)
        qrw.apply(X.ctrl(), qbit_out, wstate_ones[j])
        qrw.apply(qrout_contains_ones, node_s_ones[j], node_t_ones, qbit_out)

    for j in range(n - k):
        # check if node_s_ones[j] is present in node_t_ones and, if not, apply
        # X to w[j]
        qrw.apply(qrout_contains_zeros, node_s_zeros[j], node_t_zeros,
                  qbit_out)
        qrw.apply(X.ctrl(), qbit_out, wstate_zeros[j])
        qrw.apply(qrout_contains_zeros, node_s_zeros[j], node_t_zeros,
                  qbit_out)

    return qrw


def oracle(n, k, m, n_qubits_sum, target_value):
    qrw = QRoutineWrapper(QRoutine())
    node_s_ones = qrw.qarray_wires(k, m, "s_1", int)
    sum_reg = qrw.qarray_wires(1, n_qubits_sum, "sum", int)
    # qrout_sum = classarith.add(n_qubits_sum, m)
    qrout_sum = cuccaro_arith.adder(m, n_qubits_sum, False, False)
    with qrw.compute():
        for j in range(k):
            qrw.apply(qrout_sum, node_s_ones[j], sum_reg)
        qrw.apply(
            qi.initialize_qureg_to_complement_of_int(target_value,
                                                     n_qubits_sum, False),
            sum_reg)
    qrw.apply(Z.ctrl(n_qubits_sum - 1), sum_reg)
    qrw.uncompute()
    return qrw


def main(n,
         k,
         values: list[int],
         target_sum: int,
         low_width=True,
         to_simulate=False,
         intermediate_simulation=False):
    insert = insert_lw if low_width else insert_ld
    # Assuming no duplicates
    m = max(values).bit_length()
    print(f"Original: n {n}, k {k}, m {m}, values {values}, target sum = {target_sum}")

    values = sorted(values)
    has_repetitions = any(values[i] == values[i - 1]
                          for i in range(1, len(values)))
    if k > n / 2:
        target_sum = sum(values) - target_sum
        k = n - k
    print(f"Modified: n {n}, k {k}, m {m}, values {values}, target sum = {target_sum}")

    # the spectral gap of the johnson graph (n, k)
    delta = n / (k * (n - k))
    # 2^s >  \pi/(2 \sqrt(delta)) -> s > log_2(\pi/(2\sqrt(\delta)))
    len_s = int(round(np.log2(np.pi / (2 * np.sqrt(delta)))))
    len_s = max(1, len_s)
    # n iterations external
    n_external_iters = int(round(np.sqrt(comb(n, k))))
    n_external_iters = max(1, n_external_iters)

    # I need to store the sum of k elements, and in the worst case is the sum of the last k elements
    n_qubits_sum = sum(values[-k:]).bit_length()
    print(f"n_qubits_sum: {n_qubits_sum}; len:_s {len_s}; delta: {delta}; n_external_iters: {n_external_iters}")

    prw = ProgramWrapper(Program())
    dicke = prw.qarray_alloc(n, 1, "dicke", str)
    node_s_ones = prw.qarray_alloc(k, m, "s_1", int)
    node_s_zeros = prw.qarray_alloc(n - k, m, "s_0", int)
    node_t_ones = prw.qarray_alloc(k, m, "t_1", int)
    node_t_zeros = prw.qarray_alloc(n - k, m, "t_0", int)
    # alpha_ones = prw.qarray_alloc(1, m, "a_1", int)
    # alpha_zeros = prw.qarray_alloc(1, m, "a_0", int)
    wstate_ones = prw.qarray_alloc(k, 1, "w_1", str)
    wstate_zeros = prw.qarray_alloc(n - k, 1, "w_0", str)

    # catch all ancillae
    prw.qarray_noalloc(None,
                       None,
                       "anc",
                       wstate_zeros[-1].start + wstate_zeros[-1].length,
                       str,
                       unknown_size=True)

    qpe_s = prw.qarray_alloc(len_s, 1, "qpe_s", str)
    sum_reg = prw.qarray_alloc(1, n_qubits_sum, "sum", int)

    # dicke + bix
    prw.apply(generate(n, k), dicke)
    prw.apply(bix.bix_data_compile_time(n, m, k, values), dicke,
              node_s_ones, node_s_zeros)
    # if intermediate_simulation:
    #     print("After bix")
    #     simulate_program(prw)  # seems ok
    qrw_update = update(n, k, m, insert, has_repetitions)
    prw.apply(
        qrw_update,
        node_s_ones,
        node_s_zeros,
        node_t_ones,
        node_t_zeros,
        # alpha_ones, alpha_zeros,
        wstate_ones,
        wstate_zeros)
    if intermediate_simulation:
        print("After update")
        simulate_program(prw)

    # preparing hadamard
    for qb in qpe_s:
        prw.apply(H, qb)

    for iter_no in range(n_external_iters):
        # oracle
        qf_ora = oracle(n, k, m, n_qubits_sum, target_sum)
        # a, b -> a+b, b
        prw.apply(qf_ora, node_s_ones, sum_reg)
        if intermediate_simulation:
            print(f"Iteration {iter_no}. After oracle")
            simulate_program(prw)

        # walk
        with prw.compute():
            for qw_iter in range(len_s):
                # ref a
                prw.apply(
                    qrw_update.dag(),
                    node_s_ones,
                    node_s_zeros,
                    node_t_ones,
                    node_t_zeros,  # alpha_ones, alpha_zeros,
                    wstate_ones,
                    wstate_zeros)
                # ... ref 0^\perp
                for j in range(k):
                    prw.apply(X, wstate_ones[j])
                prw.apply(Z.ctrl(k), qpe_s[qw_iter], wstate_ones)
                for j in range(k):
                    prw.apply(X, wstate_ones[j])
                prw.apply(
                    qrw_update,
                    node_s_ones,
                    node_s_zeros,
                    node_t_ones,
                    node_t_zeros,  # alpha_ones, alpha_zeros,
                    wstate_ones,
                    wstate_zeros)
                if intermediate_simulation:
                    print(f"Iteration {iter_no}. After ref(a)")
                    simulate_program(prw)

                # ref b
                prw.apply(
                    qrw_update.dag(),
                    node_s_zeros,
                    node_s_ones,
                    node_t_zeros,
                    node_t_ones,  # alpha_zeros, alpha_ones,
                    wstate_zeros,
                    wstate_ones)
                # ... ref 0^\perp
                for j in range(n - k):
                    prw.apply(X, wstate_zeros[j])
                prw.apply(Z.ctrl(n - k), qpe_s[qw_iter], wstate_zeros)
                for j in range(k):
                    prw.apply(X, wstate_zeros[j])
                prw.apply(
                    qrw_update,
                    node_s_zeros,
                    node_s_ones,
                    node_t_zeros,
                    node_t_ones,  # alpha_zeros, alpha_ones,
                    wstate_zeros,
                    wstate_ones)
                if intermediate_simulation:
                    print(f"Iteration {iter_no}. After ref(b)")
                    simulate_program(prw)

            prw.apply(QFT(len_s).dag(), qpe_s)
            if intermediate_simulation:
                print(f"Iteration {iter_no}. After QFT")
                simulate_program(prw)
        # ref(0^\dagger)
        for j in range(len_s):
            prw.apply(X, qpe_s[j])
        if len_s > 1:
            prw.apply(Z.ctrl(len_s - 1), qpe_s)
        else:
            prw.apply(Z, qpe_s)
        for j in range(len_s):
            prw.apply(X, qpe_s[j])
        prw.uncompute()

    print("Program qubits")
    for k, v in prw._qregnames_to_properties.items():
        print(k, v.slic)

    if to_simulate:
        simulate_program(prw, qubits=[*node_s_ones])
    else:
        cr = prw.to_circ(link=[classarith, cuccaro_arith])
        print(cr.statistics())


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("two boolean params: to_simulate, and intermediate_simulation")
    to_simulate = bool(sys.argv[1])
    intermediate_simulation = bool(sys.argv[2])
    print(f"To simulate is {to_simulate}")
    values = [1, 2, 0]
    n = len(values)
    k = 1
    # m = max(values).bit_length()
    ts = 3
    main(n,
         k,
         values,
         ts,
         low_width=True,
         to_simulate=to_simulate,
         intermediate_simulation=intermediate_simulation)
