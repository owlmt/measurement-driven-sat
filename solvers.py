!pip install qiskit qiskit-aer
import math
import time
from collections import defaultdict, deque

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


# ============================================================
# Helper: normalize assignments and SAT checker
# ============================================================

def _normalize_assignment(assignment):
    """
    Convert different assignment formats to a unified list form:
        [None, True/False, True/False, ...]
    Returns None if assignment is None.
    """
    if assignment is None:
        return None

    # Already list - assume [None, bool, ...]
    if isinstance(assignment, list):
        return assignment

    # Dict format: {var_index: 0/1 or bool}
    if isinstance(assignment, dict):
        if not assignment:
            return None
        max_var = max(assignment.keys())
        lst = [None] * (max_var + 1)
        for v, val in assignment.items():
            lst[v] = bool(val)
        return lst

    raise TypeError(f"Unsupported assignment type: {type(assignment)}")


def is_satisfying(clauses, assignment):
    """
    Check whether a given assignment satisfies all clauses.

    clauses    : list of clauses, each clause is a list of ints (literals)
    assignment : list [None, True/False, ...] or dict {var: 0/1 or bool}

    Returns
    -------
    bool
        True if all clauses are satisfied under the assignment, else False.
        Returns False if assignment is None or incomplete.
    """
    ass = _normalize_assignment(assignment)
    if ass is None:
        return False

    for clause in clauses:
        clause_sat = False
        for lit in clause:
            v = abs(lit)
            if v >= len(ass):
                continue
            val = ass[v]
            if val is None:
                continue
            if (lit > 0 and val) or (lit < 0 and not val):
                clause_sat = True
                break
        if not clause_sat:
            return False

    return True


# ============================================================
# Classical baselines: DPLL and CDCL (your original code)
# ============================================================

def dpll_solve(clauses, assignment):
    # unit propagate
    changed = True
    while changed:
        changed = False
        for clause in clauses:
            unassigned = []
            satisfied = False
            for lit in clause:
                v = abs(lit)
                val = assignment[v]
                if val is None:
                    unassigned.append(lit)
                else:
                    if (lit > 0 and val) or (lit < 0 and not val):
                        satisfied = True
                        break
            if satisfied:
                continue
            if len(unassigned) == 0:
                return False
            if len(unassigned) == 1:
                lit = unassigned[0]
                assignment[abs(lit)] = (lit > 0)
                changed = True

    if all(val is not None for val in assignment[1:]):
        return True

    # Choose a variable
    v = next(i for i in range(1, len(assignment)) if assignment[i] is None)

    for val in [True, False]:
        assignment_copy = assignment[:]
        assignment_copy[v] = val
        if dpll_solve(clauses, assignment_copy):
            assignment[:] = assignment_copy
            return True

    return False


def dpll(clauses, n):
    assignment = [None] * (n + 1)
    sat = dpll_solve(clauses, n)
    return assignment if sat else None


class CDCL:
    def __init__(self, clauses, nvars):
        self.nvars = nvars
        self.clauses = clauses
        self.assign = [None] * (nvars + 1)
        self.level = [0] * (nvars + 1)
        self.reason = [None] * (nvars + 1)
        self.trail = []
        self.trail_lim = []
        self.watches = defaultdict(list)
        for c in clauses:
            self.watches[c[0]].append(c)
            if len(c) > 1:
                self.watches[c[1]].append(c)

    def value(self, lit):
        v = abs(lit)
        x = self.assign[v]
        if x is None:
            return None
        return x if lit > 0 else not x

    def propagate(self):
        q = deque([l for l in range(1, self.nvars + 1) if self.assign[l] is not None])
        while q:
            var = q.popleft()
            lit = var if self.assign[var] else -var
            for clause in self.watches[-lit]:
                if self.value(clause[0]) is True:
                    continue
                found = False
                for i in range(len(clause)):
                    if clause[i] != -lit and self.value(clause[i]) != False:
                        clause[0], clause[i] = clause[i], clause[0]
                        self.watches[clause[0]].append(clause)
                        self.watches[-lit].remove(clause)
                        found = True
                        break
                if found:
                    continue

                other = clause[1]
                val = self.value(other)
                if val is False:
                    return clause
                if val is None:
                    self.assign[abs(other)] = (other > 0)
                    self.reason[abs(other)] = clause
                    self.level[abs(other)] = len(self.trail_lim)
                    self.trail.append(other)
                    q.append(abs(other))
        return None

    def analyze(self, conflict):
        learnt = []
        seen = set()
        count = 0
        clause = conflict[:]
        lvl = len(self.trail_lim)

        while True:
            for lit in clause:
                if abs(lit) not in seen and self.level[abs(lit)] > 0:
                    seen.add(abs(lit))
                    if self.level[abs(lit)] == lvl:
                        count += 1
                    else:
                        learnt.append(lit)

            while True:
                lit = self.trail.pop()
                if abs(lit) in seen:
                    break
                self.assign[abs(lit)] = None
            count -= 1
            if count == 0:
                break
            clause = self.reason[abs(lit)]

        learnt.append(-lit)
        backtrack = 0
        for l in learnt:
            backtrack = max(backtrack, self.level[abs(l)])
        return learnt, backtrack

    def backtrack(self, level):
        while len(self.trail_lim) > level:
            start = self.trail_lim.pop()
            while len(self.trail) > start:
                lit = self.trail.pop()
                self.assign[abs(lit)] = None
                self.reason[abs(lit)] = None

    def pick_branch(self):
        for v in range(1, self.nvars + 1):
            if self.assign[v] is None:
                return v
        return None

    def solve(self):
        while True:
            conflict = self.propagate()
            if conflict:
                if len(self.trail_lim) == 0:
                    return False
                learnt, back = self.analyze(conflict)
                self.clauses.append(learnt)
                self.watches[learnt[0]].append(learnt)
                if len(learnt) > 1:
                    self.watches[learnt[1]].append(learnt)
                self.backtrack(back)
                lit = learnt[0]
                self.assign[abs(lit)] = (lit > 0)
                self.level[abs(lit)] = back
                self.reason[abs(lit)] = learnt
                self.trail_lim.append(len(self.trail))
                self.trail.append(lit)
            else:
                v = self.pick_branch()
                if v is None:
                    return True
                self.trail_lim.append(len(self.trail))
                self.assign[v] = True
                self.level[v] = len(self.trail_lim) - 1
                self.reason[v] = None
                self.trail.append(v)


# ============================================================
# Quantum part - closer to Algorithm 2 with restarts via postselection
# ============================================================

def apply_clause_projector_pi_over_2(qc, data_qubits, anc_qubit, anc_cbit, clause):
    """
    Implement one clause measurement {C_j(π/2), P_j(π/2)} in the unrotated setting.

    This corresponds to a single call to
        MEASURE(|ψ⟩, {C_j(θ), P_j(θ)})
    in the inner loop of Algorithm 2, specialized to θ = π/2.

    Steps:
      1. Map the unique violating assignment of the clause to |111...1⟩
         on the clause's data qubits.
      2. Apply multi-controlled X from those qubits onto ancilla.
      3. Undo the mapping.
      4. Measure ancilla into anc_cbit.

    Outcome:
      anc = 0  → C_j (clause passed)
      anc = 1  → P_j (clause failed)
    """
    # Ensure ancilla starts in |0⟩
    qc.reset(anc_qubit)

    # Map "literal false" to |1⟩ on the corresponding qubit
    for lit in clause:
        idx = abs(lit) - 1
        if lit > 0:
            qc.x(data_qubits[idx])

    # Multi-controlled X that fires on the violating assignment
    controls = [data_qubits[abs(lit) - 1] for lit in clause]
    qc.mcx(controls, anc_qubit)

    # Undo mapping
    for lit in clause:
        idx = abs(lit) - 1
        if lit > 0:
            qc.x(data_qubits[idx])

    # Measure ancilla to realize the projector measurement
    qc.measure(anc_qubit, anc_cbit)

    # No reset here - each clause gets its own dedicated ancilla in this construction
    # Barrier only for readability in the drawn circuit
    qc.barrier()


def build_md_circuit_with_cycles(clauses, n, cycles=1):
    """
    Build a measurement-driven SAT circuit that unrolls a fixed number
    of 'cycles' of clause checks, in the unrotated θ = π/2 setting.

    Mapping to Algorithm 2 in the paper:
      - |ψ₀⟩ = |+⟩^{⊗ n}
      - For i = 1..cycles (heuristic stand-in for r*):
            For each clause j:
                MEASURE({C_j(π/2), P_j(π/2)})

    We assign one dedicated ancilla + classical bit to each clause in each cycle.
    That way, for each shot we can see which clause checks failed and
    postselect on those shots where all ancilla bits are 0.

    Classical bit layout:
      - First we add c_anc (size = cycles * m).
      - Later we add c_out (size = n) for the final data measurement.
      - Qiskit bitstrings are then [meas bits][anc bits] when spaces are removed.
    """
    m = len(clauses)
    num_anc = cycles * m

    # Data qubits and ancilla qubits
    q = QuantumRegister(n, 'x')
    anc = QuantumRegister(num_anc, 'anc')

    # Classical bits for ancilla outcomes
    c_anc = ClassicalRegister(num_anc, 'c_anc')

    qc = QuantumCircuit(q, anc, c_anc)

    # Initial state |+⟩^{⊗ n}
    qc.h(q)

    anc_idx = 0
    for _ in range(cycles):
        for clause in clauses:
            apply_clause_projector_pi_over_2(qc, q, anc[anc_idx], c_anc[anc_idx], clause)
            anc_idx += 1

    # Final readout of data qubits
    c_out = ClassicalRegister(n, 'meas')
    qc.add_register(c_out)
    qc.measure(q, c_out)

    return qc, num_anc


def quantum_sat_solver(clauses, n, cycles=1, max_trials=128):
    """
    Quantum SAT solver with 'restarts' implemented by postselection
    in the unrotated θ = π/2 setting.
    """
    backend = AerSimulator()
    qc, num_anc = build_md_circuit_with_cycles(clauses, n, cycles=cycles)

    # Use lowest optimization level to avoid long transpile times
    tqc = transpile(qc, backend, optimization_level=0)

    result = backend.run(tqc, shots=max_trials).result()
    counts = result.get_counts()

    success_counts = {}
    total_success_shots = 0

    for bitstring, cnt in counts.items():
        s = bitstring.replace(" ", "")
        rev = s[::-1]

        anc_bits = rev[0:num_anc]
        data_bits = rev[num_anc:num_anc + n]

        # Restart logic: discard shots where any ancilla = 1
        if any(b == '1' for b in anc_bits):
            continue

        total_success_shots += cnt
        success_counts[data_bits] = success_counts.get(data_bits, 0) + cnt

    if total_success_shots == 0:
        assignment = [None] * (n + 1)
        return assignment, 0.0

    best_data_bits = max(success_counts, key=success_counts.get)

    assignment = [None] * (n + 1)
    for i, b in enumerate(best_data_bits):
        assignment[i + 1] = (b == '1')

    success_rate = total_success_shots / float(max_trials)
    return assignment, success_rate


def benchmark(clauses, n, cycles=1, max_trials=128):
    print("===== BENCHMARK =====")

    # DPLL
    t0 = time.time()
    dpll_assignment = [None] * (n + 1)
    sat = dpll_solve(clauses, dpll_assignment)
    dpll_sol = dpll_assignment if sat else None
    t_dpll = time.time() - t0
    dpll_sat = is_satisfying(clauses, dpll_sol)

    # CDCL
    t0 = time.time()
    cdcl = CDCL(clauses[:], n)
    cdcl.solve()
    cdcl_sol = cdcl.assign
    t_cdcl = time.time() - t0
    cdcl_sat = is_satisfying(clauses, cdcl_sol)

    # Quantum
    t0 = time.time()
    qsol, q_success_rate = quantum_sat_solver(clauses, n, cycles=cycles, max_trials=max_trials)
    t_q = time.time() - t0
    q_sat = is_satisfying(clauses, qsol)

    print("DPLL:    ", dpll_sol, "SAT?:", dpll_sat, "time:", t_dpll)
    print("CDCL:    ", cdcl_sol, "SAT?:", cdcl_sat, "time:", t_cdcl)
    print("Quantum: ", qsol, "SAT?:", q_sat, "success_rate:", q_success_rate, "time:", t_q)


# ============================================================
# Example instance and run
# ============================================================

if __name__ == "__main__":
    clauses = [
        [1, -2, 3],
        [-1, 2, 4],
        [3, -4, 5],
        [-3, 4, -5],
        [1, 6, -7],
        [-1, -6, 7],
        [2, -8, 9],
        [-2, 8, -9],
        [3, 10, -1],
        [-3, -10, 1],
        [4, -5, 6],
        [-4, 5, -6],
        [7, -8, 9],
        [-7, 8, -9],
        [6, 10, -2],
        [-6, -10, 2]
    ]
    n = 10

    # Example: 1 cycle per trial, arbitrary number of trials (shots) e.g 1024 0r 512
    benchmark(clauses, n, cycles=1, max_trials=10)
