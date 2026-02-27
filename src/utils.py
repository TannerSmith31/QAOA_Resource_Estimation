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
