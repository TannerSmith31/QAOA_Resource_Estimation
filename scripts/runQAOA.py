import argparse
from qaoaExperiment import QAOAExperiment, loadExperiment
from src.utils import genGraph
import os

def main():
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


if __name__ == "__main__":
    main()