###############################################################################
#
#
#
###############################################################################
import numpy as np
import os, csv
import matplotlib.pyplot as plt
from numba import types
from numba.typed import Dict
from scipy.signal import find_peaks

def read_from_swc(filename):
    """Get astrocyte morphology from a text file (.txt) in the swc format.

    Each line in the text file must contain the following data space-separated:
    index, compartment type, x coordinate, y coordinate z coordinate, radius,
    index of parent node. So, each line must contain seven entries.

    x coordinate, y coordinate, z coordinate and radius must be in m. For the
    astrocyte models, comparment type is 0 for soma and 1 for processes. The
    parent index for the somatic compartment must be -1.

    Example for a 9-comparment morphology:
    1 0 0 0 0 20e-6 -1
    2 1 21e-6 0 0 2e-6 1
    3 1 22e-6 0 0 2e-6 2
    4 1 23e-6 0 0 2e-6 3
    5 1 24e-6 0 0 1e-6 4
    6 1 25e-6 0 0 0.5e-6 5
    7 1 26e-6 0 0 0.25e-6 6
    8 1 27e-6 0 0 0.125e-6 7
    9 1 28e-6 0 0 0.0625e-6 8

    Parameters
    ----------
    filename: str
        path to the morphology text file

    Return
    ------
    points: list of tuples
        loaded morphology from the text file
    """
    with open(filename,'r') as f:
        points = []
        
        for line_n, line in enumerate(f):
            
            if line.startswith('#') or len(line) == 0:
                continue
                
            splitted = line.split()
            
            if len(splitted) != 7:
                raise ValueError((f"Each line of an SWC file has to contain "
                                 f"7 space-separated entries, but line "
                                 f"{line_n + 1} contains {len(splitted)}."))
                
            index, comp_type, x, y, z, radius, parent = splitted
            
            points.append((int(index),
                          int(comp_type), 
                          float(x), 
                          float(y),
                          float(z), 
                          float(radius), 
                          int(parent)))
            
        return points

def calculate_morphological_parameters(points):
    """Calculate morpjological parameters from a specified astrocyte morphology. The
    somatic compartment is a sphere and the other comparments are cylinders.

    Let r be the radius of the somatic compartments, so its surface area, cross sectional
    areav and volume are calculate as follow:
    Surface Area = 4*pi*r^2 
    Cross Sectional Area = 4*pi*r^2 (flux from inside to outside of the sphere)
    Volume = 4/3*pi*r^3

    For the cylindrical compartments with radius r, the lenth, surface area, cross 
    sectional area and volume are calculate as:
    Length = euclidean distance between two compartments
    Surface Area = 2*pi*r*length 
    Cross Sectional Area = pi*r^2
    Volume = pi*r^2*length

    The cytosol-endoplasmic reticulum volume ratio is calculated as (Patrushev et al.
    2013):
    ratio_ER 0.15*exp(-(0.073e-6 * Surface Area/Volume)**2.34) 
    
    References
    ----------
    Patrushev I, Gavrilov N, Turlapov V, Semyanov A. Subcellular location of
    astrocytic calcium stores favors extrasynaptic neuron–astrocyte communication.
    Cell Calcium. 2013; 54:343-9

    Parameters
    ----------
    points: list of list or tuple (n x 7),
        list with the morphologycal parameter of each compartment (in swc format).

    Return
    ------
    morp_params: numpy array.
        Each columm represent:
            0 - Compartment type (0 for soma and 1 for process)
            1 - Compartment radius
            2 - Compartment length (radius for soma)
            3 - Surface area
            4 - Cross sectional area (surface area for soma)
            5 - Volume
            6 - Ratio ER
    """
    n_compart = len(points)
    morph_params = np.zeros(shape=(n_compart, 7))
    
    
    for i, compart in enumerate(points):
                
        if compart[-1] == -1:
            morph_params[i][0] = 0                                                          # Compartment Type (0 for soma)
            morph_params[i][1] = compart[-2]                                                # Radius
            morph_params[i][2] = compart[-2]                                                # Lenght (equal the radius for soma)
            
            morph_params[i][3] = 4*np.pi*compart[-2]**2                                     # Sphere surface area
            morph_params[i][4] = 4*np.pi*compart[-2]**2                                     # Sphere cross section area
            morph_params[i][5] = 4/3*np.pi*compart[-2]**3                                   # Sphere volume
            
        else: 
            morph_params[i][0] = 1                                                          # Compartment Type
            morph_params[i][1] = compart[-2]                                                # Radius
            
            # If the parent compartment is the soma, the lenght of the compartment is calculated considering the radius of the soma
            if compart[-1] == 1:
                morph_params[i][2] = np.sqrt(compart[2]**2 + compart[3]**2 + compart[4]**2) - points[0][-2]      # Length
            else:
                morph_params[i][2] = np.sqrt((compart[2] - points[compart[-1] - 1][2])**2 + 
                                            (compart[3] - points[compart[-1] - 1][3])**2 +
                                            (compart[4] - points[compart[-1] - 1][4])**2)      # Length
            
            morph_params[i][3] = 2*np.pi* morph_params[i][1]*morph_params[i][2]             # Cylinder surface area
            morph_params[i][4] = np.pi*morph_params[i][1]**2                                # Cylinder cross section area
            morph_params[i][5] = np.pi*compart[-2]**2*morph_params[i][2]                    # Cylinder volume
            
        morph_params[i][6] = 0.15*np.exp(-(0.073e-6 * morph_params[i][3]/morph_params[i][5])**2.34)  # Ratio ER
        
    return morph_params

def build_connection_matrix(points):
    """Construct the connection matrix representing the astrocyte compartments
    connections. Each entry indicates a connected (= 1) or disconnected (= 0)
    compartment.
    
    Parameters
    ----------
    points: list of list or tuple (n x 7),
        list with the morphologycal parameter of each compartment (in swc format).
    
    Return
    ------
    connection_matrix: numpy array (number of compartments x number of compartments)
        matrix indicating compartment connections
    """
    n = len(points)
    connection_matrix = np.zeros(shape=(n,n))
    
    for i_connection, connection in enumerate(points):
        if connection[-1] != -1:
            connection_matrix[connection[0] - 1,connection[-1] - 1] = 1
            connection_matrix[connection[-1] - 1,connection[0] - 1] = 1
        
    return connection_matrix

def create_numba_dictionary(dictionary):
    """Create the numba type dictionary. Native Python dictionaries are not 
    compatible with numba.

    Parameters
    ----------
    dictionary: Python dictionary.

    Return
    ------
    result_dict: Numba dictionary
    """
    result_dict = Dict.empty(key_type = types.unicode_type,
                                       value_type = types.float64)

    for key in dictionary.keys():
        result_dict[key] = dictionary[key]
        
    return result_dict

def bissec_method(f, a, b, tol = 1e-10, **kwargs):
    """Calculate the roots of the function f in the interval [a,b] with a tolerance
    tol using the bisection method.

    Parameters
    ----------
    f: function
        function for which the root will be calculated.
    a: float
        lower bound for the interval in which to search the function root.
    b: float
        upper bound for the interval in which to search the function root.
    tol: float
        error accepted for calculating the root.
    **kwargs: keyword-arguments for the function f.

    Return
    ------
    Approximated root
    """
    
    while (b - a >= tol):
        if f(a, **kwargs)*f(0.5*(b + a), **kwargs) < 0:            
            b = 0.5*(b + a)
        else:
            a = 0.5*(b + a)
            
    return 0.5*(b + a)