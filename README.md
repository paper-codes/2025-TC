# 2025-TC

Reference implementation for the paper:

> G. Lancellotti, S. Perriello, A. Barenghi, G. Pelosi.
> *Solving the Subset Sum Problem via Quantum Walk Search.*
> IEEE Transactions on Computers, vol. 75, n. 1, 2026.
> [DOI: 10.1109/TC.2025.3625044](https://doi.org/10.1109/TC.2025.3625044)

A gate-level implementation of an MNRS quantum walk for the Subset Sum Problem
on Johnson graphs, built on top of the [myQLM](https://myqlm.github.io/)
framework. All shared quantum subroutines are part of the
[`qat-utils`](https://github.com/tigerjack/qat-utils) package.

## Requirements

- [myQLM](https://myqlm.github.io/)
- [`qat-utils`](https://github.com/tigerjack/qat-utils)

## Usage

Run the test suite:

```shell
REVERSIBLE_ON=1 pytest -s
```

Run `cssp.py` on the built-in instance (`n=3`, `k=1`, `values=[0,1,2]`, `target=1`):

```shell
python cssp.py True True
```

The two positional arguments are flags: `to_simulate` runs the full
state-vector simulation, `intermediate_simulation` prints the state after each
major sub-circuit.

## Citation

If you use this code in academic work, please cite the paper above.
