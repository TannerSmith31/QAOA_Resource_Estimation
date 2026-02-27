from qiskit_aer import AerSimulator
from qiskit import transpile
import networkx as nx
import os
import time
import json
from networkx.readwrite import json_graph #TODO: figure out why this isnt in the graph file
import numpy as np 
from utils import jsonify
from graphs.py import sanitize_for_json  #TODO: rename sanitize_for_json to sanitizeGraphForJson
from qaoa.py import getEnergyExtremes, gen_QEC_noise_model, generateQAOACircuitTemplate, solveQAOA



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
    
######################### END OF EXPERIMENTS CLASS ####################

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