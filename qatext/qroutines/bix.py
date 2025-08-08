import logging
from ctypes import ArgumentError
from typing import List

from qat.lang.AQASM.gates import SWAP, X
from qat.lang.AQASM.misc import build_gate
from qat.lang.AQASM.routines import QRoutine
from qatext.qroutines import qregs_init
from qatext.qroutines.arith import adder
from qatext.qroutines.qubitshuffle import rotate
from qatext.utils.bits.conversion import get_bitarray_from_int

LOGGER = logging.getLogger(__name__)


@build_gate(
    "BIX_IDXS", [int, int, bool], lambda n, _, x: n + n *
    (n.bit_length() if x else (n - 1).bit_length()))
def bix_indexes_compile_time(n: int, weight: int, idx_start_at_one: bool):
    """Given a bitstring of length `n`, having exactly `weight` qubits set to
    1, store into `weight` registers the indexes of the 1's of the bitstring,
    and `n - weight` registers the weight of the 0's of the bitstring. If
    `idx_start_at_one` is True, the result will be a 0-indexed array.

    It should be applied to the following registers:
    - qreg of length `n`, containing `weight` 1's
    - `weight` qregs, each of size `log2(n)`
    - `n - weight` qregs, each of size `log2(n)`

    It also uses ancillary registers:
    - `qreg1s_add`, of size bit_length(n), containing a temporary register for the `weight` qregs, used if `weight` > 1
    - `qreg0s_add`, of size bit_length(n), containing a temporary register for the `n-weight` qregs, used if `n-weight ` > 1
    - `const` register, of size bit_length(n), containing the fixed value `1`
    All ancillary registers are cleaned at the end of the circuit.

    Internally, it invokes left rotate circuit and addition circuits; last one
    is abstract and must be specialized.

    """

    if weight < 1 or weight >= n:
        raise ArgumentError("Weight should be >=1 and < n, given {}" % weight)
    LOGGER.debug("weight %d", weight)
    qrout = QRoutine()
    add = 1 if idx_start_at_one else 0
    m = (n - 1 + add).bit_length()
    LOGGER.debug("m %d", m)

    wreg = qrout.new_wires(n)
    oregs = []
    zregs = []
    for i in range(weight):
        oregs.append(qrout.new_wires(m))
    for i in range(n - weight):
        zregs.append(qrout.new_wires(m))

    ancillae1 = qrout.new_wires(m)
    qrout.set_ancillae(ancillae1)
    oregs.append(ancillae1)
    ancillae2 = qrout.new_wires(m)
    qrout.set_ancillae(ancillae2)
    zregs.append(ancillae2)
    # the register that will hold the constants +1 and -n
    const = qrout.new_wires(m)
    qrout.set_ancillae(const)

    #
    qset1 = qregs_init.initialize_qureg_given_int(1, m, little_endian=False)
    qadd = adder(m, m, False, False)
    qxor = qregs_init.copy_register(m)
    qleftrotones = rotate.reg_reversal(len(oregs), m, 1)
    qleftrotzeros = rotate.reg_reversal(len(zregs), m, 1)
    final_clean = n if idx_start_at_one else n - 1
    qsetfinal = qregs_init.initialize_qureg_given_int(final_clean,
                                                      m,
                                                      little_endian=False)

    qrout.apply(qset1, const)
    for i in range(n):
        if i != 0 or (i == 0 and idx_start_at_one):
            qrout.apply(qadd, const, oregs[0])
            qrout.apply(qadd, const, zregs[0])

        # if wreg[i] is 1, we left rotate the ones
        # if weight > 1:
        qrout.apply(qleftrotones.ctrl(1), wreg[i], *oregs)
        # ... and add to the ones register
        qrout.apply(qxor.ctrl(1), wreg[i], oregs[-1], oregs[0])

        # ...otw, we left rotate the zeros
        qrout.apply(X, wreg[i])
        qrout.apply(qleftrotzeros.ctrl(1), wreg[i], *zregs)
        qrout.apply(qxor.ctrl(1), wreg[i], zregs[-1], zregs[0])
        qrout.apply(X, wreg[i])

    # reset const register to 0
    qrout.apply(qset1.dag(), const)

    # set it to value n
    qrout.apply(qsetfinal, const)
    for qreg in (oregs[0], zregs[0]):
        # The topmost register, qreg, should be decreased by the constant value
        # n, stored in the the register const. However, when we use the
        # sub(qreg, const) circuit, the result is stored in const.
        #
        # Additionally, note that val(qreg) = n + delta, delta >= 0; i.e., the
        # topmost register is always greater than n.
        #
        # So first we swap the two qregs; now val(qreg) = n; val(const) = n + delta
        for qb1, qb2 in zip(qreg, const):
            qrout.apply(SWAP, qb1, qb2)
        # then we negate const; val(const) = complement(n+delta)
        for qb in const:
            qrout.apply(X, qb)
        # then, we add to it the constant register and complement again, obtaining
        # val(const) = delta
        qrout.apply(qadd, qreg, const)
        for qb in const:
            qrout.apply(X, qb)
        # then, we switch again: val(qreg) = delta; val(const) = n
        for qb1, qb2 in zip(qreg, const):
            qrout.apply(SWAP, qb1, qb2)
    # reset const register to 0
    qrout.apply(qsetfinal.dag(), const)

    # if weight == 1 or weight == n-1:
    # there is an extra register
    qrout.apply(qleftrotzeros, *zregs)
    qrout.apply(qleftrotones, *oregs)

    return qrout


@build_gate("BIX_DATAD_DIFF", [int, int, int, List], lambda n, m, w, x: n + n * m)
def bix_data_diff_compile_time(n: int, m: int, weight: int, elems: List):
    """Given a bitstring of length `n`, having exactly `weight` qubits set to
    1, store into `weight` registers the values `elems[i]` if `dicke[i] == 1`,
    and `n - weight` registers the values `elems[i]` if `dicke[i] == 0`.

    It should be applied to the following registers:
    - qreg_dicke: the register containing the dicke state

    It uses an additional ancilla register, reset to all zeros after
    - one qreg of size `log2(n)`
    If `weight` is equal to 1 or n-1, it uses an additional support array

    Internally, it invokes left rotate circuit and addition circuits; last one
    is abstract and must be specialized.

    """

    if weight < 1 or weight >= n:
        raise ArgumentError("Weight should be >=1 and < n, given {}" % weight)
    elems_diffs = [elems[0]] + [j - i for i, j in zip(elems, elems[1:])]

    qrout = QRoutine()
    wreg = qrout.new_wires(n)
    oregs = []
    zregs = []
    for i in range(weight):
        oregs.append(qrout.new_wires(m))
    for i in range(n - weight):
        zregs.append(qrout.new_wires(m))
    ancillae1 = qrout.new_wires(m)
    qrout.set_ancillae(ancillae1)
    oregs.append(ancillae1)
    ancillae2 = qrout.new_wires(m)
    qrout.set_ancillae(ancillae2)
    zregs.append(ancillae2)
    # the register that will hold the constants +1 and -n
    # in theory can be smaller than this
    const = qrout.new_wires(m)
    qrout.set_ancillae(const)

    #
    qadd = adder(m, m, False, False)
    qxor = qregs_init.copy_register(m)
    qleftrotones = rotate.reg_reversal(len(oregs), m, 1)
    qleftrotzeros = rotate.reg_reversal(len(zregs), m, 1)

    # _otmp = [0] * (weight+1)
    # _ztmp = [0] * (n-weight+1)

    for i in range(n):
        qset1 = qregs_init.initialize_qureg_given_int(elems_diffs[i],
                                                      m,
                                                      little_endian=False)
        qrout.apply(qset1, const)
        if elems_diffs[i] != 0:
            qrout.apply(qadd, const, oregs[0])
            qrout.apply(qadd, const, zregs[0])

        # if wreg[i] is 1, we left rotate the ones
        qrout.apply(qleftrotones.ctrl(1), wreg[i], *oregs)
        # ... and copy to the first register
        qrout.apply(qxor.ctrl(1), wreg[i], oregs[-1], oregs[0])

        # ...otw, we left rotate the zeros
        qrout.apply(X, wreg[i])
        qrout.apply(qleftrotzeros.ctrl(1), wreg[i], *zregs)
        qrout.apply(qxor.ctrl(1), wreg[i], zregs[-1], zregs[0])
        qrout.apply(X, wreg[i])
        # reset const register to 0
        qrout.apply(qset1.dag(), const)

    # final_clean = n if idx_start_at_one else n - 1
    final_clean = elems[-1]
    # set it to value n
    qsetfinal = qregs_init.initialize_qureg_given_int(final_clean,
                                                      m,
                                                      little_endian=False)
    qrout.apply(qsetfinal, const)
    for qreg in (oregs[0], zregs[0]):
        # The topmost register, qreg, should be decreased by the constant value
        # n, stored in the the register const. However, when we use the
        # sub(qreg, const) circuit, the result is stored in const.
        #
        # Additionally, note that val(qreg) = n + delta, delta >= 0; i.e., the
        # topmost register is always greater than n.
        #
        # So first we swap the two qregs; now val(qreg) = n; val(const) = n + delta
        for qb1, qb2 in zip(qreg, const):
            qrout.apply(SWAP, qb1, qb2)
        # then we negate const; val(const) = complement(n+delta)
        for qb in const:
            qrout.apply(X, qb)
        # then, we add to it the constant register and complement again, obtaining
        # val(const) = delta
        qrout.apply(qadd, qreg, const)
        for qb in const:
            qrout.apply(X, qb)
        # then, we switch again: val(qreg) = delta; val(const) = n
        for qb1, qb2 in zip(qreg, const):
            qrout.apply(SWAP, qb1, qb2)
    # reset const register to 0
    qrout.apply(qsetfinal.dag(), const)

    # if weight == 1 or weight == n-1:
    # there is an extra register
    qrout.apply(qleftrotzeros, *zregs)
    qrout.apply(qleftrotones, *oregs)

    return qrout

@build_gate("BIX_DATA", [int, int, int, List], lambda n, m, w, x: n + n * m)
def bix_data_compile_time(n: int, m: int, weight: int, elems: List):
    """Given a bitstring of length `n`, having exactly `weight` qubits set to
    1, store into `weight` registers the values `elems[i]` if `dicke[i] == 1`,
    and `n - weight` registers the values `elems[i]` if `dicke[i] == 0`.

    It should be applied to the following registers:
    - qreg_dicke: the register containing the dicke state
    - qreg_ones: the register that will contain the `weight` element for which the corresponding indexes is 1
    - qreg_zeros: the register that will contain the `weight` element for which the corresponding indexes is 0

    Internally, it invokes left rotate circuit and addition circuits; last one
    is abstract and must be specialized.

    """
    # main difference with the _diff one is that it works directly on the data,
    # not using additional ancillae for the diff

    if weight < 1 or weight >= n:
        raise ArgumentError("Weight should be >=1 and < n, given {}" % weight)

    qrout = QRoutine()
    wreg = qrout.new_wires(n)
    oregs = []
    zregs = []
    for i in range(weight):
        oregs.append(qrout.new_wires(m))
    for i in range(n - weight):
        zregs.append(qrout.new_wires(m))

    #
    qleftrotones = rotate.reg_reversal(len(oregs), m, 1)
    qleftrotzeros = rotate.reg_reversal(len(zregs), m, 1)

    for i in range(n):
        # copy the first element
        qrout_init = qregs_init.initialize_qureg_given_int(elems[i], m, False)
        qrout.apply(qrout_init.ctrl(1), wreg[i], oregs[0])
        if weight > 1:
            # if wreg[i] is 1, we left rotate the ones
            qrout.apply(qleftrotones.ctrl(1), wreg[i], *oregs)

        # ...otw, we left rotate the zeros
        qrout.apply(X, wreg[i])
        qrout.apply(qrout_init.ctrl(1), wreg[i], zregs[0])
        if n - weight > 1:
            # if wreg[i] is 0, we left rotate the zeros
            qrout.apply(qleftrotzeros.ctrl(1), wreg[i], *zregs)
        qrout.apply(X, wreg[i])


    return qrout

@build_gate("BIX_MATRIX", [int, int, int, int, List],
            lambda n, r, m, w, x: n * r * m + n)
def bix_matrix_compile_time(n: int, columns: int, m: int, weight: int,
                            matrix: List):
    """It is given a bitstring of length `n`, having exactly `weight` qubits
    set to 1. The goal is to store into `weight` registers (each one composed
    by `columns` cells, each cell having `m` bits) the submatrix composed by
    the rows of `matrix` indexed by the `i` bits set to `1` of the bitstring,
    and on `n-weight` the remaining ones. The matrix has size `n X columns`
    size, and its flattened row-wise (e.g. [[0, 1, 2], [3, 4, 5]] is [0, 1, 2,
    3, 4, 5])


    It should be applied to the following registers:
    - qreg_dicke: the register containing the dicke state
    - qreg_omatrix
    - qreg_zmatrix

    It uses additional ancillary register, reset to all zeros after
    """
    # This one is the VBE proposed to TC, which works with direct encoding
    # instead of deltas

    if weight < 1 or weight >= n:
        raise ArgumentError("Weight should be >=1 and < n, given {}" % weight)
    # elems_diffs = [elems[0]] + [j - i for i, j in zip(elems, elems[1:])]
    rows = n

    qrout = QRoutine()
    wreg = qrout.new_wires(n)

    omatrix_flat = []
    # flattened row-wise
    for row in range(weight):
        for col in range(columns):
            qr = qrout.new_wires(m)
            omatrix_flat.append(qr)
    zmatrix_flat = []
    for row in range(n - weight):
        for col in range(columns):
            qr = qrout.new_wires(m)
            zmatrix_flat.append(qr)
    LOGGER.debug("omatrix_flat, len %d", len(omatrix_flat))
    # LOGGER.debug(omatrix_flat)
    LOGGER.debug("zmatrix_flat, len %d", len(zmatrix_flat))
    # LOGGER.debug(zmatrix_flat)

    #
    qleftrotones = rotate.reg_reversal(len(omatrix_flat), m, columns)
    qleftrotzeros = rotate.reg_reversal(len(zmatrix_flat), m, columns)

    for row in range(rows):
        for col in range(columns):
            matrix_val = matrix[row * columns + col]
            LOGGER.debug("matrix[%d][%d]", row, col)
            LOGGER.debug("It is computed as row*columns + col")
            val = get_bitarray_from_int(matrix_val, m, False)
            LOGGER.debug("It is %d, meaning %s", matrix_val, val)
            q_row_init = qregs_init.initialize_qureg_given_bitarray(val, False)
            LOGGER.debug("Initialize omatrix[%d] (%s) to %s", col,
                         str(omatrix_flat[col]), val)
            qrout.apply(q_row_init.ctrl(1), wreg[row], omatrix_flat[0 + col])
            qrout.apply(X, wreg[row])
            qrout.apply(q_row_init.ctrl(1), wreg[row], zmatrix_flat[0 + col])
            qrout.apply(X, wreg[row])
        if weight != 1:
            LOGGER.debug("Rotating ones ctrld on wreg[%d]", row)
            qrout.apply(qleftrotones.ctrl(1), wreg[row], omatrix_flat)
        if n - weight != 1:
            LOGGER.debug("Rotating zeros ctrld on wreg[%d]", row)
            qrout.apply(X, wreg[row])
            qrout.apply(qleftrotzeros.ctrl(1), wreg[row], zmatrix_flat)
            qrout.apply(X, wreg[row])

    return qrout
