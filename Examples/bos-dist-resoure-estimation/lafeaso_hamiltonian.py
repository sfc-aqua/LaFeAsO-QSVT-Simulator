import random
import pickle
import numpy as np
import cirq

from openfermion import (
    FermionOperator,
    QubitOperator,
    jordan_wigner
)

from pyLIQTR.ProblemInstances.ProblemInstance import ProblemInstance
from pyLIQTR.utils.resource_analysis import estimate_resources


# ----------------------------------------------------------------------
#  Hamiltonian Construction
# ----------------------------------------------------------------------

def build_lafeaso_effective_hamiltonian(n_cells: int = 1, qubits_per_cell: int = 20) -> FermionOperator:
    """
    Construct a synthetic effective Hamiltonian for LaFeAsO using a simplified
    random-parameter parameterization. Each crystal cell contains a fixed number
    of spin–orbitals, with a variety of on-site, hopping, Coulomb and exchange
    interactions.

    Parameters
    ----------
    n_cells : int
        Number of identical repeated unit cells.
    qubits_per_cell : int
        Number of spin–orbitals per cell.

    Returns
    -------
    FermionOperator
        The total fermionic Hamiltonian.
    """
    
    H = FermionOperator()

    for cell in range(n_cells):
        offset = cell * qubits_per_cell
        spin_orb = list(range(offset, offset + qubits_per_cell))

        # 1. On-site energies
        for p in spin_orb:
            H += FermionOperator(f"{p}^ {p}", random.uniform(-1.5, -0.5))

        # 2. Hopping terms
        for _ in range(712):
            p, q = random.sample(spin_orb, 2)
            coeff = random.uniform(-0.5, 0.5)
            H += FermionOperator(f"{p}^ {q}", coeff)
            H += FermionOperator(f"{q}^ {p}", coeff)  # Hermitian conjugate

        # 3. On-site Coulomb (paired orbitals)
        for i in range(0, qubits_per_cell, 2):
            H += FermionOperator(
                f"{offset + i}^ {offset + i} {offset + i + 1}^ {offset + i + 1}",
                random.uniform(2.0, 4.0),
            )

        # 4. Long-range Coulomb
        for _ in range(3680):
            p, q = random.sample(spin_orb, 2)
            H += FermionOperator(
                f"{p}^ {p} {q}^ {q}", random.uniform(0.1, 0.5)
            )

        # 5. Exchange interactions
        for _ in range(40):
            p, q = random.sample(spin_orb, 2)
            H += FermionOperator(
                f"{p}^ {q}^ {q} {p}", random.uniform(0.05, 0.2)
            )

        # 6–7. Pair hopping and correlated hopping
        for _ in range(40):
            p, q, r, s = random.sample(spin_orb, 4)
            H += FermionOperator(
                f"{p}^ {q}^ {s} {r}", random.uniform(0.05, 0.2)
            )

    return H



# ----------------------------------------------------------------------
#  ProblemInstance definition
# ----------------------------------------------------------------------

class LaFeAsOHamiltonian(ProblemInstance):
    """
    Encapsulation of the LaFeAsO effective Hamiltonian as a PyLIQTR problem
    instance. Provides convenient access to fermionic and qubit representations.
    """

    def __init__(self, n_cells: int = 1, qubits_per_cell: int = 20):
        super().__init__()
        self.n_cells = n_cells
        self.qubits_per_cell = qubits_per_cell

        self._n_qubits = n_cells * qubits_per_cell
        self.ferm_op = build_lafeaso_effective_hamiltonian(
            n_cells=n_cells, qubits_per_cell=qubits_per_cell
        )

    def get_fermionic_operator(self) -> FermionOperator:
        return self.ferm_op

    def get_qubit_hamiltonian(self) -> QubitOperator:
        return jordan_wigner(self.ferm_op)

    def get_number_qubits(self) -> int:
        return self._n_qubits



# ----------------------------------------------------------------------
#  Circuit Construction
# ----------------------------------------------------------------------

def build_cirq_circuit_from_qubit_operator(qubit_op: QubitOperator, t: float = 1.0) -> cirq.Circuit:
    """
    Build a Cirq Trotterized circuit for exp(-i H t), where H is provided as a
    QubitOperator.

    Parameters
    ----------
    qubit_op : QubitOperator
        Hamiltonian expressed in qubit form.
    t : float
        Evolution time.

    Returns
    -------
    cirq.Circuit
        A circuit implementing exp(-i H t).
    """

    # Determine qubit count
    max_idx = max((q for term in qubit_op.terms for q, _ in term), default=-1)
    qubits = [cirq.LineQubit(i) for i in range(max_idx + 1)]

    circuit = cirq.Circuit()

    for term, coeff in qubit_op.terms.items():
        if not term or abs(coeff) < 1e-12:
            continue

        pauli_ops = []
        for q, p in term:
            pauli_ops.append({
                "X": cirq.X,
                "Y": cirq.Y,
                "Z": cirq.Z
            }[p](qubits[q]))

        pauli_string = cirq.PauliString(*pauli_ops)

        # Trotterized evolution exp(-i coeff t P)
        theta = -coeff.real * t / np.pi
        circuit += pauli_string ** theta

    return circuit



# ----------------------------------------------------------------------
#  Main execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    N_cells = 100
    qubits_per_cell = 20

    # Build Hamiltonian
    ferm_op = build_lafeaso_effective_hamiltonian(N_cells, qubits_per_cell)
    qubit_op = jordan_wigner(ferm_op)

    # Build circuit
    circuit = build_cirq_circuit_from_qubit_operator(qubit_op, t=1.0)

    print(circuit)

    # Resource estimation
    resources = estimate_resources(circuit, circuit_precision=1e-10, profile=True)
    print("\nResource Estimate:\n", resources)

    # Save
    out_name = f"Cells_{N_cells}_Qubits_{qubits_per_cell}_lafeaso_circuit.pkl"
    with open(out_name, "wb") as f:
        pickle.dump(circuit, f)
