from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit import QuantumCircuit, transpile
from pygridsynth.gridsynth import gridsynth_gates
from scipy.optimize import minimize
from itertools import product
import mpmath
import numpy as np 
from qiskit.circuit import Parameter # needed for parametric circuits 



#########################################
#    CIRCUIT FEATURE FUNCTIONS  (noise & decomposition)
#########################################

"""
Function to generate a noisy model to run circuits on where the noise generated is based on the code distance simulated
PARAMS:
    p_phys = noise of machine without QEC
    p_th = threshold noise value
    d = code distance to be simulated 
"""
def gen_QEC_noise_model(p_phys: float, p_th: float, d: int):
    # --- Compute logical error rate ---
    if d == 0:
        p_L = p_phys  #no error correction
    else:
        exponent = (d + 1) / 2
        p_L = 0.03 * (p_phys / p_th)**exponent

    # --- Build noise model ---
    noise_model = NoiseModel()

    ### ONE QUBIT ERROR
    # Symmetric logical Pauli channel
    logical_error = pauli_error([
        ('X', p_L / 3),
        ('Y', p_L / 3),
        ('Z', p_L / 3),
        ('I', 1 - p_L)
    ])

    # Apply to all logical single- and two-qubit gates
    all_gates = ['x', 'y', 'z', 'h', 'sx', 'id']
    noise_model.add_all_qubit_quantum_error(logical_error, all_gates)
    
    ## 2 QUBIT ERROR
    # All 15 non-identity two-qubit Paulis
    paulis_2q = []
    single = ["I", "X", "Y", "Z"]

    for p1 in single:
        for p2 in single:
            op = p1 + p2
            if op != "II":
                paulis_2q.append(op)

    prob_per_term = p_L / len(paulis_2q)

    two_qubit_channel = [(op, prob_per_term) for op in paulis_2q]
    two_qubit_channel.append(("II", 1 - p_L))

    two_qubit_error = pauli_error(two_qubit_channel)

    two_qubit_gates = ["cx", "cz", "swap"]

    noise_model.add_all_qubit_quantum_error(two_qubit_error, two_qubit_gates)

    return noise_model

"""
 Function to decompose a quantum circuit to a native gateset + rz and then 
 apply gridsynth to rz gates in a quantum circuit (qc) with a given precision (precision)
"""
def transpile_circuit_with_gridsynth(qc, precision=mpmath.mpf("1e-5")):
    # We include Rz because that's what we will feed to GridSynth
    basis_gates = ['h', 's', 't', 'cx', 'rz']
    
    # Transpile the circuit to simplify everything down to these components
    basis_transpiled_qc = transpile(qc, basis_gates=basis_gates, optimization_level=3)
    
    transpiledQC = QuantumCircuit(qc.num_qubits)
    
    for instruction in basis_transpiled_qc.data:
        gate = instruction.operation
        qargs = instruction.qubits
        
        if gate.name == 'rz':
            angle = gate.params[0]
            target_angle = mpmath.mpf(str(angle)) # Convert to high-precision string for GridSynth

            #1: get sequence (e.g. "SHTHT..."). NOTE: GridSynth returns sequences in matrix order (right-to-left) so we must reverse it for circuit order (left-to-right)
            gate_sequence = gridsynth_gates(target_angle, precision)[::-1]

            #2: map characters to Qiskit gates
            for char in gate_sequence:
                if char == 'T':
                    transpiledQC.t(qargs[0])
                elif char == 'H':
                    transpiledQC.h(qargs[0])
                elif char == 'S':
                    transpiledQC.s(qargs[0])
                elif char == 'X':
                    transpiledQC.x(qargs[0])
                #GridSynth may output 'I' or phases (like global phase 'W'), skip if not a gate
            
        else:
            #For all other gates (CNOT, H, etc.) just copy them over
            transpiledQC.append(gate, qargs)

    return transpiledQC

#########################################
#    QAOA FUNCTIONS
#########################################

"""
Cost function for Ising Energy
function to compute the objective value for a given x    [f(z)= sum(h_i*z_i)+sum(sum(J_ij*z_i*z_j))]
 INPUT:
 G = graph encoding h and J coefficients
 x = bitstring of assigned values for each z_i (in our case x_i IN (0,1) -> z_i IN (-1,1))
 OUTPUT:
 f = the cost that we are going to want to minimize with the optimizer
"""
def calcCost(G,x):
    f = 0.0
    s = [1 if bit == '1' else -1 for bit in x]

    for i, data in G.nodes(data=True):
        f += data.get("weight", 0.0) * s[i]

    for i, j, data in G.edges(data=True):
        f += data.get("weight", 0.0) * s[i] * s[j]

    return f

"""
function to get the expected value based on a number of shots
"""
def getExpectedVal(G, counts):
    total = sum(counts.values())
    exp_val = 0.0
    for bitstring, freq in counts.items():
        f = calcCost(G, bitstring[::-1])  # Qiskit reverses bit order
        exp_val += f * freq
    return exp_val / total

"""
function to get the true minimum energy from coefficients h & J over bitstrings x
"""
def getEnergyExtremes(G):
    n = G.number_of_nodes()
    all_bitstrings = [''.join(bits) for bits in product('01', repeat=n)]
    min_energy = min(calcCost(G, x) for x in all_bitstrings)
    max_energy = max(calcCost(G, x) for x in all_bitstrings)
    return min_energy, max_energy

## function to generate QAOA circuit template that parameter values will be bound to
# G = graph encoding the QAOA problem (h & J as node and edge weights)
# numLayers = the number of layers the circuit will have
def generateQAOACircuitTemplate(G, numLayers):
    numNodes = G.number_of_nodes()
    
    #Create parameter objects
    betas = [Parameter(f"beta_{i}") for i in range(numLayers)]
    gammas = [Parameter(f"gamma_{i}") for i in range(numLayers)]
    
    pqc = QuantumCircuit(numNodes)
    pqc.h(range(numNodes))
    
    for layer in range(numLayers):
        for i, j, data in G.edges(data=True):
            J_ij = data.get("weight", 0.0)
            pqc.cx(i,j)
            pqc.rz(2 * gammas[layer] * J_ij, j)
            pqc.cx(i,j)
        for i, data in G.nodes(data=True):
            h_i = data.get("weight", 0.0)
            pqc.rz(2 * gammas[layer] * h_i, i)
        #MIXER
        for i in range(numNodes):
            pqc.rx(2 * betas[layer], i)
            
    pqc.measure_all()
    return pqc, betas, gammas
    
#Function to solve QAOA with given parameters
# INPUT:
#  G = the graph encoding the QAOA problem (h and J coeffs as weights of nodes and edges)
#  backend - the backend we want to run the QAOA on (NOTE: will have to pass in AerSimulator(noise_model=<NOISE MODEL>) or <Ideal Model>)
#  numLayers = number of layers in the circuit
def solveQAOA(G, backend, numLayers, maxIters = 1000, numShots = 4096):
    param_qc, beta_params, gamma_params = generateQAOACircuitTemplate(G, numLayers)
    transpiled_base = transpile(param_qc, backend)
    
    #objective function that will be minimized. It essentially takes in the alpha and beta params and
    #runs the QAOA circuit and returns the achieved energy (expected value)
    def objective(params):
        p_val = len(params)//2
        beta_vals, gamma_vals = params[:p_val], params[p_val:]
        
        #create map beta_i -> value, gamma_i -> value
        bind_dict={}
        for i in range(numLayers):
            bind_dict[beta_params[i]] = beta_vals[i]
            bind_dict[gamma_params[i]] = gamma_vals[i]
            
        #bind the param values to the circuit params
        bound_circuit = transpiled_base.assign_parameters(bind_dict)
        
        result = backend.run(bound_circuit, shots = numShots).result()
        counts = result.get_counts()
        exp_val = getExpectedVal(G, counts) 
        return exp_val #minimize energy
    
    ### Classical optimization
    initial_guess = np.random.uniform(0, np.pi, 2 * numLayers) #initialize to random array [0,pi]
    res = minimize(objective, initial_guess, method="COBYLA", options={'maxiter': maxIters, 'disp': False})
    
    optimized_params = res.x
    optimized_betas = optimized_params[:numLayers]
    optimized_gammas = optimized_params[numLayers:]
    achieved_energy = float(res.fun)
    
    # Compute approximation ratio
    min_energy, max_energy = getEnergyExtremes(G)
    AR = (max_energy - achieved_energy) / (max_energy - min_energy)
    
    #store results in dictionary
    results = {}
    results["optimized_params"] = optimized_params
    results["achieved_energy"] = achieved_energy
    results["AR"] = AR

    return results