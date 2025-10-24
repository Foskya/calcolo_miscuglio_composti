import numpy as np
from scipy.optimize import linprog

def parameters():
    compounds = { # element espressed in %
        #                Na     Cl     Ca     K      N      Fe     S      Mg     P      C
        'CaCO3':        [0    , 0    , 0.400, 0    , 0    , 0    , 0    , 0    , 0    , 0.120],
        'Ca(NO3)2':     [0    , 0    , 0.244, 0    , 0.171, 0    , 0    , 0    , 0    , 0    ],
        'MgCl2x6H2O':   [0    , 0.349, 0    , 0    , 0    , 0    , 0    , 0.120, 0    , 0    ],
        #'NH3':          [0    , 0    , 0    , 0    , 0.822, 0    , 0    , 0    , 0    , 0    ],
        'MgSO4':        [0    , 0    , 0    , 0    , 0    , 0    , 0.266, 0.202, 0    , 0    ],
        #'CH4N2O':       [0    , 0    , 0    , 0    , 0.466, 0    , 0    , 0    , 0    , 0.200],
        '(NH4)2HPO4':   [0    , 0    , 0    , 0    , 0.212, 0    , 0    , 0    , 0.235, 0    ],
        'K2SO4':        [0    , 0    , 0    , 0.449, 0    , 0    , 0.184, 0    , 0    , 0    ],
        'CH3COOH':      [0    , 0    , 0    , 0    , 0    , 0    , 0    , 0    , 0    , 0.400],
        'NaOH':         [0.575, 0    , 0    , 0    , 0    , 0    , 0    , 0    , 0    , 0    ],
        'NaCl':         [0.393, 0.607, 0    , 0    , 0    , 0    , 0    , 0    , 0    , 0    ],
        'CitricoC6H8O': [0    , 0    , 0    , 0    , 0    , 0    , 0    , 0    , 0    , 0.375]
        }
    element_matrix = np.array(list(compounds.values())).T # linprog needs elements as row and compounds as columns, so I transposte the matrix
    elements =          ['Na',  'Cl',  'Ca',  'K',   'N',   'Fe',  'S',   'Mg',  'P',   'C']
    target_composition =[0.257, 0.028, 0.000, 0.030, 0.019, 0.000, 0.010, 0.001, 0.004, 0.109]
    tolerance = 0.0071
    cost_coeficent = np.zeros(len(compounds))
    bounds = [(0, None) for _ in range(len(compounds))] # constraint: 0 imply non negative bounds and None implies no upper bound
    return compounds, element_matrix, elements, target_composition, tolerance, cost_coeficent, bounds

def optimization_problem(compounds, element_matrix, target_composition, tolerance, cost_coeficent, bounds):
    A_ub = np.vstack([element_matrix, -element_matrix])
    b_ub = np.hstack([np.array(target_composition) + tolerance, -(np.array(target_composition) - tolerance)])
    A_eq = np.ones((1, len(compounds))) # to make the sum of all the % equal to 1
    b_eq = np.array([1])
    result = linprog(c=cost_coeficent, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return result

def main():
    compounds, element_matrix, elements, target_composition, tolerance, cost_coeficent, bounds = parameters()
    result = optimization_problem(compounds, element_matrix, target_composition, tolerance, cost_coeficent, bounds)
    if result.success:
        proportions = result.x
        print("\nDISTRIBUZIONE COMPOSTI:")
        for i, name in enumerate(compounds.keys()):
            print(f"- {name}: {proportions[i] * 100:.2f}%")
        print("\nELEMENTI CHIMICI:")
        print(f"Tolleranza: {tolerance}")
        final_composition = element_matrix @ proportions
        for i, el in enumerate(elements):
            print(f"- {el}: {final_composition[i] * 100:.2f}% (Target: {target_composition[i] * 100:.2f}%)")
    else:
        print("Non è stata trovata nessuna soluzione valida")
        print("Considera l'aumento della tolleranza")

main()
