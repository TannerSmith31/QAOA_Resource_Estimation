### IMPORTS ###
import numpy as np 
import networkx as nx
import copy 
import time
from itertools import product
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit.visualization import plot_histogram
from qiskit.circuit import Parameter # needed for parametric circuits 
import json #needed for file stuff for exporting experiment data
from networkx.readwrite import json_graph
import argparse
import os
import sys

## GRIDSYNTH IMPORTS
import mpmath
from pygridsynth.gridsynth import gridsynth_gates
import numpy as np

"""Function to jsonify any value"""
def jsonify(v):
    # NumPy scalar → Python scalar
    if isinstance(v, np.generic):
        return v.item()

    # NumPy array → Python list (recursively)
    if isinstance(v, np.ndarray):
        return [jsonify(x) for x in v.tolist()]

    # Python list/tuple → recursively convert elements
    if isinstance(v, (list, tuple)):
        return [jsonify(x) for x in v]

    # Python dict → recursively convert values
    if isinstance(v, dict):
        return {k: jsonify(x) for k, x in v.items()}

    return v  # already safe

####################################
#   GRAPH FUNCTIONS
####################################

"""
function to generate graphs with different topology strategies and weight generation strategies
topologyStrategy
   0 = 3-regular
   1 = random/Erdos-Eenyi
   2 = Fully Connected
   3 = Barabasi-Albert (power-law) with preferential attachment m=1
weightStrategy
   0 = Random Choice w_ij = {-1,+1} uniformly
   1 = Uniform: w_ij ~ U[-1,1]
   2 = Gaussian: w_ij ~ N(0,1)
"""
def genGraph(numNodes, topologyStrategy, weightStrategy, weighNodes = False):
    G = nx.Graph()  #generate empty undirected graph
    
    if topologyStrategy<0 or topologyStrategy>3 or weightStrategy<0 or weightStrategy>2:
        print("topologyStrategy(" + str(topologyStrategy) + ") or weightStrategy(" + str(weightStrategy) + ") out of bounds")
        print("returning empty graph")
        return G  #return an empty graph
    
    ## Generate list of edges based on topologyStrategy
    if topologyStrategy == 0:   #3-regular
        G = nx.random_regular_graph(d=3, n=numNodes)  #NOTE: 3-regular graphs must have even number of nodes
    
    if topologyStrategy == 1:  #random/Erdos-Eenyi
        edgeProb = 0.5
        G = nx.erdos_renyi_graph(numNodes, edgeProb)
        
    if topologyStrategy == 2:  #Fully Connected
        G = nx.complete_graph(numNodes)
        
    if topologyStrategy == 3:  #Barabasi-Albert (power-law)
        G = nx.barabasi_albert_graph(numNodes, 1)
        
    ## assign weights to edges based on weightStrategy
    for u,v in G.edges():
        if weightStrategy == 0:    #Random Choice
            G[u][v]['weight'] = np.random.choice([-1,1])

        if weightStrategy == 1:    #Uniform Dist
            G[u][v]['weight'] = np.random.uniform(low=-1, high=1)

        if weightStrategy == 2:    #Gaussian Dist
            G[u][v]['weight'] = np.random.normal(0,1)
    
    ## assign weights to nodes if needed
    if weighNodes == True:
        #give nodes weight of -1 or +1 randomly
        for i in G.nodes():
            G.nodes[i]["weight"] = np.random.choice([-1, 1])
    else:
        #assign all nodes weight of 0
        for i in G.nodes():
            G.nodes[i]["weight"] = 0.0
    
    return G

"""
Function to display a graph with its edges and verticies labeled
"""
def drawGraph(G, showNodeWeight=False):
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=500)

    # --- Draw node labels ---
    if showNodeWeight:
        #Show weight AND label
        node_labels = {i: G.nodes[i].get('weight', 0) for i in G.nodes()}
    else:
        #Only show label
        node_labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='black')

    # --- Draw edge labels for weights ---
    edge_labels = {(u, v): round(d.get('weight', 0.0), 2) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()
    return

"""
Function to make all graph attributes json serializable
"""
def sanitize_for_json(G):
    H = G.copy()

    # Fix node attributes
    for _, attrs in H.nodes(data=True):
        for k, v in attrs.items():
            attrs[k] = jsonify(v)

    # Fix edge attributes
    for _, _, attrs in H.edges(data=True):
        for k, v in attrs.items():
            attrs[k] = jsonify(v)

    # Fix graph-level attributes
    for k, v in H.graph.items():
        H.graph[k] = jsonify(v)

    return H


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
        
####################################
#  EXPERIMENT AND ALPINE FUNCTIONS
####################################

"""
CLASS to store information about one experimental run of QAOA. An experiment is a certain graph topology, weighting, and number of nodes
PARAMETERS:
    #INPUT PARAMS
    problem_graph: the graph that will be used for this QAOA cost function (h & J obtained from node and edge weights)
    graph_topology: the topology of the graph for this experiment
    graph_weighting: the weighting strategy for the graph of this experiment
    num_shots: the number of shots to use in the QAOA
    num_restarts: the number of times to restart QAOA with random params to try to get optimal AR
    
    #OTHER OBJECT PARAMS
    num_nodes: number of nodes in the problem graph
    min_energy: the minimum energy of all assignments of the QAOA problem graph
    max_energy: the maximum energy of all assignments of the QAOA problem graph
    
    #OUTPUT PARAMS (will be stored in self.results)
    optimized_betas: optimum betas
    optimized_gammas: optimized gammas
    achieved_energy: energy achieved by running the QAOA
    runtime: time it took to get the solution
    AR: approximation ratio of achievedEnergy/minEnergy
    circuit_depth: depth of the generated circuit
"""
class QAOAExperiment:
    def __init__(self, problem_graph, graph_topology, graph_weighting, num_shots=1000, num_restarts=5):        
        self.graph_topology = graph_topology
        self.graph_weighting = graph_weighting
        self.num_nodes = problem_graph.number_of_nodes()
        self.num_shots = num_shots
        self.num_restarts = num_restarts
        
        self.problem_graph = problem_graph
        
        ## get max and min energy from problem graph
        min_energy, max_energy = getEnergyExtremes(self.problem_graph)
        self.min_energy = min_energy
        self.max_energy = max_energy
    
        self.results = {}  # Stores information from the result of the experiment
        
    """
    Runs this Experiment in QAOA and stores results
    PARAMS:
        maxLayers: the maximum number of layers that will be run if multiple layers are to be run
        maxCodeDistance: the maximum code distance the experiment will run to if multiple code distances are to be run
        layersToRun (array): specify what layers you want to run (i.e. pass in [1,2] and it will run with 1 layer and 2 layers)
            DEFAULT: if nothing is passed in, it will run QAOA for all layers 0 -> max layers
        codeDistance (array): specify what code distances you want to run it on (i.e. pass in [-1,3,5] to run with ideal sim, d=3, and d=5)
            NOTE: -1 for ideal sim, 0 for noisy sim with no EC
            DEFAULT: if nothing is passed in, it will run QAOA for ideal sim and all (odd) distances (excluding 1) up to maxCodeDistance (ideal, d=0, d=3, d=5, d=7, ...)
        exportFilepath: if a filepath is passed in, the experiment will be exported to that filepath after each code distance is run, otherwise it is not exported
        uniqueFileId: an integer that will be appended to the filename if exporting (to ensure unique filenames)
    """
    def run(self,maxLayers = 10, maxCodeDistance = 13, layersToRun = [], distancesToRun = [], exportFilepath = None, uniqueFileId = 0):
        
        ### Fill distances to run with DEFAULT if nothing passed (ideal,0,3,5,...,maxCodeDistance)
        if len(distancesToRun) == 0: 
            distancesToRun = [-1, 0] + list(range(3, maxCodeDistance + 1, 2))
        
        ### Fill layers to run with DEFAULT if nothing is passed (1,2,...,maxLayers)
        if len(layersToRun) == 0:
            layersToRun = range(1,maxLayers+1)
        
        ### Go through each layer and code distance combo and run
        for codeDistance in distancesToRun:
            for numLayers in layersToRun:
                print(f"Running Experiment n{self.num_nodes}T{self.graph_topology}W{self.graph_weighting}:d{codeDistance}L{numLayers}")
                
                ## Create Backend
                threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) #TODO: REMOVE if broke. ADD TO VS CODE IF NOT
                if codeDistance == -1: #code distance -1 means use ideal sim
                    backend=AerSimulator(max_parallel_threads=threads, max_parallel_shots=threads)  #TODO: REMOVE params if broke. ADD TO VS CODE IF NOT
                else:
                    noiseModel=gen_QEC_noise_model(p_phys=0.003, p_th=0.0057, d=codeDistance)
                    backend=AerSimulator(noise_model=noiseModel, max_parallel_threads=threads, max_parallel_shots=threads)  #TODO: remove parallel params if broke. ADD TO VS CODE IF NOT
                    
                ##get an estimate for the circuit depth based on the problem graph and backend 
                dummyQC, dummyBetas, dummyGammas = generateQAOACircuitTemplate(self.problem_graph,numLayers)
                transpiledCircuit = transpile(dummyQC, backend)
                self.store_result(codeDistance, numLayers, "circuit_depth", transpiledCircuit.depth())
                
                #store start time for runtime calculation
                start_time = time.time()
        
                #run num_restarts times and take the best AR to avoid local minimums
                self.store_result(codeDistance, numLayers, "AR", 0)
                for i in range(0,self.num_restarts):   #randomly restart num_restarts times and store best to avoid local minimums
                    results = solveQAOA(self.problem_graph, backend, numLayers, maxIters=1000, numShots=self.num_shots)
                    if results["AR"] > self.retrieve_result(codeDistance, numLayers, "AR"):
                        self.store_result(codeDistance, numLayers, "AR", results["AR"])
                        self.store_result(codeDistance, numLayers, "achieved_energy", results["achieved_energy"])
                        self.store_result(codeDistance, numLayers, "optimized_betas", (results["optimized_params"])[:numLayers])
                        self.store_result(codeDistance, numLayers, "optimized_gammas", (results["optimized_params"])[numLayers:])
        
                #calculate runtime
                end_time = time.time()
                self.store_result(codeDistance, numLayers, "runtime", end_time - start_time)  #TODO: may need to convert units?
            
            #save data to file after every distance run if exportFilepath specified
            if exportFilepath is not None:
                self.exportExperiment(filepath=exportFilepath, uniqueFileId=uniqueFileId)
        
        print("All Experiments successful")


    """Stores results in the results dict"""
    def store_result(self, codeDistance, numLayers, key, value):
        codeDistanceKey = f"d{codeDistance}"
        numLayersKey = f"l{numLayers}"

        # Create the nested structure if needed
        if codeDistanceKey not in self.results:
            self.results[codeDistanceKey] = {}

        if numLayersKey not in self.results[codeDistanceKey]:
            self.results[codeDistanceKey][numLayersKey] = {}

        # Store the result
        self.results[codeDistanceKey][numLayersKey][key] = value
    
    
    """Retrieves a result in the results dict"""
    def retrieve_result(self, codeDistance, numLayers, key):
        codeDistanceKey = f"d{codeDistance}"
        numLayersKey = f"l{numLayers}"

        if codeDistanceKey not in self.results:
            raise KeyError(f"No results stored for code distance '{codeDistanceKey}'")

        if numLayersKey not in self.results[codeDistanceKey]:
            raise KeyError(f"No results stored for layer '{numLayersKey}' under code distance '{codeDistanceKey}'")

        if key not in self.results[codeDistanceKey][numLayersKey]:
            raise KeyError(f"Key '{key}' not found for {codeDistanceKey}, {numLayersKey}")

        return self.results[codeDistanceKey][numLayersKey][key]
    
    """Prints a summary of the experiment results to the console"""
    def summary(self):
        print("===== QAOA Experiment Summary =====")

        # Basic graph info
        print(f"Number of nodes: {self.num_nodes}, Number of edges: {self.problem_graph.number_of_edges()}")
        print(f"Graph topology: {self.graph_topology}, Weight Strategy: {self.graph_weighting}")
        print(f"maxEnergy: {self.max_energy}, minEnergy: {self.min_energy}")

        # Results overview
        if not self.results:
            print("No results stored yet.")
            return

        print("\nResults by code distance and layer:")

        # Only show these result fields
        SHOWABLE_FIELDS = {"AR", "runtime", "circuit_depth"}

        for codeDistanceKey, layerDict in self.results.items():
            print(f"   {codeDistanceKey}:")
            if not layerDict:
                print("       (no layers stored)")
                continue

            for layerKey, resultDict in layerDict.items():
                print(f"       {layerKey}:")

                if not resultDict:
                    print("           (empty)")
                    continue

                # Filter values by allowed keys
                for k, v in resultDict.items():
                    if k in SHOWABLE_FIELDS:
                        print(f"           {k}: {v}")

        print("===================================")

         
    """Function to generate a filename based on this experiment"""
    def genFilename(self, uniqueId):
        simType = "Noisy"
        filename = f"n{self.num_nodes}T{self.graph_topology}W{self.graph_weighting}_Id{uniqueId}"
        return filename
        
    """Exports data in parameters to a file (parameter 'filename' is either a name of a file or a filepath)"""
    def exportExperiment(self, filename=None, filepath=None, uniqueFileId = 0):
        
        if filename == None:  #generate filename if none provided
            filename = self.genFilename(uniqueFileId)
        
        if filepath is not None: #if filepath given, join it with the filename
            if len(filepath) != 0: #if the filepath is "", file will be put in cwd so no need to make new directory
                os.makedirs(filepath, exist_ok=True) #ensure directory exists
            full_path = os.path.join(filepath, filename)

        else:  #if filepath NOT given, just use current working directory
            full_path = filename
            
        #numpy lists and dicts with integer or tuple keys are not JSON serializable
        #this code converts numpy lists so results is JSON serializable
        json_safe_results = None
        json_safe_problem_graph = nx.node_link_data(sanitize_for_json(self.problem_graph))   # convert to serializable dict
        try:
            if self.results != None:
                json_safe_results = {key: jsonify(value) for key, value in self.results.items()}
        except Exception as e:
            raise RuntimeError(f"Error converting data to JSON-safe format: {e}")
            
        
        #create a dictionary of all experiment data that will be turned to a JSON object
        try:
            data = {
                "graph_topology": self.graph_topology,
                "graph_weighting": self.graph_weighting,
                "num_nodes": self.num_nodes,
                "num_shots": self.num_shots,
                "num_restarts": self.num_restarts,
                
                "problem_graph": json_safe_problem_graph,
                "min_energy": self.min_energy,
                "max_energy": self.max_energy,

                "results": json_safe_results,
            }
        except AttributeError as e:
            raise AttributeError(f"Missing expected attribute when exporting data: {e}")
        
        # Save to JSON
        try:
            with open(full_path, "w") as f:
                json.dump(data, f, indent=2)
        except (IOError, OSError) as e:
            raise IOError(f"Failed to write export file '{full_path}': {e}")
        except TypeError as e:
            raise TypeError(f"Data contained non-serializable values: {e}")
        
        print(f"Data Exported to {full_path} successfully")

    """
    Calculates the weighted average relative approximation ratio of each distance in this experiment. Calculated by
      [SUM_l in layers (f(l) * (noisyAR/idealAR))]/ SUM_l in layers (f(l))
    where f(l) = 1 if l<L and f(l) = L / l
    where L = min required layer for n nodes

    Returns a dictionary with keys=distances, values = WARAR for that distance
    """
    def calcWeightedAvgRelativeAR(self):
        #dictionary with keys as distances and values as sum (f(l)*AR_n(l)/AR_i(l)) for each l
        WARARs= {}
        AR_desired = 0.9
        
        #subfunction to calculate l_ideal (the minimum l value which AR(l) >= AR_desired)
        #if no layer achieves AR_desired, it returns 1 layer beyond the highest tested
        def find_L_ideal(AR_desired):
            min_ideal_L = np.inf
            max_L = -np.inf
            for layerKey,resultsDict in self.results["d-1"].items():
                L = int(layerKey[1:]) #layerKeys are of the form l<layerNum> so parse layerNum as int
                max_L = max(max_L, L)
#                 print(f"minL: {min_ideal_L}  l: {L}  resultDictVal:{resultsDict['AR']}")
                if resultsDict["AR"] >= AR_desired:
                    min_ideal_L = min(min_ideal_L, L)
            # If we found at least one layer that meets AR_desired
            if min_ideal_L != np.inf:
                return min_ideal_L

            # Otherwise return largest layer + 1
            return max_L + 1
        
        #subfunction for the weight equation
        def f(l):
            w = 0.999
            d = 3
    
            n=self.num_nodes
            L_th = np.ceil((w*np.log10(n)/np.log10(d/np.log(2)))/2)
            L_ideal = find_L_ideal(AR_desired)
            S = -(pow(L_th-L_ideal,2)) / np.log(0.1)
            print(f"L_th: {L_th} L_ideal:{L_ideal} S:{S}")
            if l < L_th:
                return 0
            else:
                return np.exp(-np.pow(l-L_ideal,2) / S)                          


        for distKey,layersDict in self.results.items():
            cur_WARAR = 0  #initialize WARAR to 0 for dist
            cur_sum_fx = 0  #initialize sum_fx to 0 for dist

            for layerKey,resultsDict in layersDict.items():
                idealLayerAR = self.results["d-1"][layerKey]["AR"]
                noisyLayerAR = self.results[distKey][layerKey]["AR"]
                layerInt = int(layerKey[1:]) #layerKeys are of the form l<layerNum> so parse layerNum as int
                print(f"d:{distKey} f(l):{f(layerInt)} l:{layerInt} idealAR:{idealLayerAR} noisyAR:{noisyLayerAR}")  #FOR DEBUGGING ONLY. TODO: remove when done
                cur_WARAR += f(layerInt) * (noisyLayerAR/idealLayerAR)
                cur_sum_fx += f(layerInt)

            WARARs[distKey] = cur_WARAR / cur_sum_fx

        return WARARs
    

    ############################## END OF EXPERIMENT CLASS ###################################
            
"""Function to load an experiment object from a file"""
def loadExperiment(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Cannot load experiment because file does not exist: {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)

    # Reconstruct the problem graph
    problem_graph = json_graph.node_link_graph(data["problem_graph"])

    # Reconstruct results (already JSON-safe, no conversion needed)
    results = data["results"]

    # Extract metadata
    graph_topology = data["graph_topology"]
    graph_weighting = data["graph_weighting"]
    num_nodes = data["num_nodes"]
    num_shots = data["num_shots"]
    num_restarts = data["num_restarts"]
    

    # Create a new QAOAExperiment instance (assuming you can pass all these)
    exp = QAOAExperiment(
        problem_graph = problem_graph,
        graph_topology=graph_topology,
        graph_weighting=graph_weighting,
        num_shots=num_shots,
        num_restarts=num_restarts,
    )

    #set the results
    exp.results = results

    return exp

###################################################
# MAIN SCRIPT
###################################################
if __name__ == "__main__":
    #Parse command line arguments
    parser = argparse.ArgumentParser(description="Run QAOA experiment on a graph")
    
    # Graph generation mode
    parser.add_argument('--numNodes', type=int, help="Number of nodes for new graph")
    parser.add_argument('--weightID', type=int, help="Weight strategy for new graph (0=random choice[-1,1],1=uniform dist[-1,1],2=gaussian dist[-1,1]")
    # Existing experiment mode
    parser.add_argument('--inputFile', type=str, help="Path to existing experiment JSON")
    parser.add_argument('--layersToRun', type=int, nargs='+', help="List of layers to run")
    parser.add_argument('--distancesToRun', type=int, nargs='+', help="List of code distances to run")
    # Common
    parser.add_argument('--outputDir', type=str, default='./results', help="Directory to save results")
    
    
    args = parser.parse_args() 
    #args = parser.parse_args(args=[]) if 'ipykernel' in sys.argv[0] else parser.parse_args() #UNCOMMENT IF WANT TO TEST IN JUPYTER
    
    #ensure output directory exists
    os.makedirs(args.outputDir, exist_ok=True)
    
    ### EXPERIMENT CODE ###
    # MODE 1: load experiment from file and build upon it
    if args.inputFile is not None:
        print(f"Loading experiment from {args.inputFile}")
        exp = loadExperiment(args.inputFile)
        layersToRun = args.layersToRun
        distancesToRun = args.distancesToRun
        
    # MODE 2: create new experiment with numNodes and weightID
    elif args.numNodes is not None and args.weightID is not None:
        print(f"Generating new graph: N{args.numNodes}W{args.weightID}")
        gTopology = 0    #only doing 3 regular for now
        if args.layersToRun is not None:
            layersToRun = args.layersToRun
        else:
            layersToRun = [1,2,3,4,5,7,9]  #default layers to run 
        if args.distancesToRun is not None:
            distancesToRun = args.distancesToRun
        else:
            distancesToRun = [-1,0,7,9,13,15] #defualt code distances to run

        G = genGraph(args.numNodes, gTopology, args.weightID)
        exp = QAOAExperiment(
            problem_graph=G,
            graph_topology=0,  #Only doing 3-regular for now
            graph_weighting= args.weightID,
            num_shots=4096,
            num_restarts=5,
        )
    #MODE ERROR
    else:
        raise ValueError("Must provide either --inputFile or both --numNodes and --weightID")
    
    #run experiment and export it
    uniqueFileId = 2
    exp.run(maxLayers = 10, maxCodeDistance = 17, layersToRun = layersToRun, distancesToRun = distancesToRun, exportFilepath = args.outputDir, uniqueFileId = uniqueFileId)
    exp.exportExperiment(filepath=args.outputDir,uniqueFileId = uniqueFileId)
    
    print("Experiment finished successfully")