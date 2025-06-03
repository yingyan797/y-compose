import numpy as np
import os, subprocess
from ltlf_tools.parser.ltlf import LTLfParser
from ltlf_tools.ltlf2dfa import MonaProgram, ter2symb, simplify_guard, symbols, output2dot

CDIR = os.getcwd()  # .../y-compose/
MONA_PATH = f"{CDIR}/project/ltlf_tools/mona.exe"

def parse_dfa(p_formula, dfa_text):
    """
    Parse a DFA description and extract its components.
    
    Returns:
        dict: A dictionary containing:
            - free_variables: List of free variables
            - initial_state: The initial state (int)
            - accepting_states: Set of accepting states (ints)
            - rejecting_states: Set of rejecting states (ints)
            - transitions: List of tuples (state_i, condition, state_j)
    """
    lines = dfa_text.strip().split('\n')
    result = {
        'formula': p_formula,
        'free_variables': [],
        'initial_state': None,
        'accepting_states': set(),
        'rejecting_states': set(),
        'transitions': [],
        'matrix': []
    }

    dot_trans = {}
    free_variables = []
    min_state, max_state = None, None
    # Process each line
    for line in lines:
        line = line.strip()
        
        # Extract free variables
        if line.startswith("DFA for formula with free variables:"):
            var_part = line.split(":", 1)[1].strip()
            result['free_variables'] = var_part.split()
            free_variables = symbols(
                tuple(fv.lower() for fv in result["free_variables"])
            )
        
        # Extract initial state
        elif line.startswith("Initial state:"):
            result['initial_state'] = int(line.split(":", 1)[1].strip())
        
        # Extract accepting states
        elif line.startswith("Accepting states:"):
            states_str = line.split(":", 1)[1].strip()
            result['accepting_states'] = [int(s) for s in states_str.split()]
        
        # Extract rejecting states
        elif line.startswith("Rejecting states:"):
            states_str = line.split(":", 1)[1].strip()
            result['rejecting_states'] = [int(s) for s in states_str.split()]
        
        # Extract transitions
        elif line.startswith("State "):
            # Format: "State X: COND -> state Y"
            parts = line.split(":", 1)
            state_i = int(parts[0].replace("State ", "").strip())
            
            # Parse the transition part
            transition_part = parts[1].strip()
            condition, target = transition_part.split("->")
            condition = condition.strip()
            state_j = int(target.replace("state", "").strip())

            if state_i == 0 and state_j == 1:
                continue            
            result['transitions'].append((state_i, condition, state_j))

            if free_variables:
                guard = ter2symb(free_variables, condition)
            else:
                guard = ter2symb(free_variables, "X")

            low = min(state_i, state_j)
            up = max(state_i, state_j)
            if min_state is None or low < min_state:
                min_state = low
            if max_state is None or up > min_state:
                max_state = up
        
            if (state_i, state_j) in dot_trans.keys():
                dot_trans[(state_i, state_j)].append(guard)
            else:
                dot_trans[(state_i, state_j)] = [guard]
    min_state = max(1, min_state)
    matrix = [["" for _ in range(min_state, max_state+1)] for _ in range(min_state, max_state+1)]
    mat_repr = [["" for _ in range(min_state, max_state+1)] for _ in range(min_state, max_state+1)]
    for c, guards in dot_trans.items():
        simplified_guard = simplify_guard(guards)
        matrix[c[0]-1][c[1]-1] = simplified_guard
        mat_repr[c[0]-1][c[1]-1] = str(simplified_guard).lower()
    result["matrix"] = mat_repr
    return result, matrix

def formula_to_dfa(ifml, file_name):
    parser = LTLfParser()
    formula = parser(ifml)       # returns an LTLfFormula
    prog = MonaProgram(formula).mona_program()
    ipath = os.path.join(CDIR, "project", "static", "mona_files", f"{file_name}.mona")
    opath = os.path.join(CDIR, "project", "static", "dfa_files", f"{file_name}.dfa")
    try:
        with open(ipath, "w+") as file:
            file.write(prog)
            lines = prog.split("\n")
            p_formula = lines[0][1:-1]
            mona_in = lines[1:-1]
    except IOError:
        print("[ERROR]: Problem opening the mona file!")
    cmd = f'{MONA_PATH} -q -u -w {ipath} > {opath}'
    if os.system(cmd) == 0:
        with open(opath, "r") as f:
            mona_output = f.read()
        return parse_dfa(p_formula, mona_output), (mona_in, mona_output) 
    return [False], mona_in

if __name__ == "__main__":
    parser = LTLfParser()
    formula = parser("!a | !b")
    # print(formula)
    print(formula.to_nnf())
