import numpy as np
import os, subprocess
from ltlf2dfa.parser.ltlf import LTLfParser
from ltlf2dfa.ltlf2dfa import output2dot, MonaProgram

CDIR = os.getcwd()
MONA_PATH = os.path.join(CDIR, "refer_code", "mona.exe")

def parse_dfa(dfa_text):
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
        'free_variables': [],
        'initial_state': None,
        'accepting_states': set(),
        'rejecting_states': set(),
        'transitions': []
    }
    
    # Process each line
    for line in lines:
        line = line.strip()
        
        # Extract free variables
        if line.startswith("DFA for formula with free variables:"):
            var_part = line.split(":", 1)[1].strip()
            result['free_variables'] = var_part.split()
        
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
            
            result['transitions'].append((state_i, condition, state_j))
    
    return result

def formula_to_dfa(ifml, mona_name, dfa_name, final_graph=False):
    parser = LTLfParser()
    formula = parser(ifml)       # returns an LTLfFormula
    prog = MonaProgram(formula).mona_program()
    ipath = os.path.join(CDIR, "project", "static", "mona_files", f"{mona_name}.mona")
    opath = os.path.join(CDIR, "project", "static", "dfa_files", f"{dfa_name}.dfa")
    try:
        with open(ipath, "w+") as file:
            file.write(prog)
    except IOError:
        print("[ERROR]: Problem opening the mona file!")
    cmd = f'{MONA_PATH} -q -u -w {ipath} > {opath}'
    if os.system(cmd) == 0:
        with open(opath, "r") as f:
            mona_output = f.read()
        if final_graph:
            return output2dot(mona_output)
       
        return parse_dfa(mona_output)
    return False

if __name__ == "__main__":
    print(formula_to_dfa("(a U b) & (c U d)", "and_until", "and_until"))
