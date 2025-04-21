import numpy as np
import os, subprocess
from ltlf2dfa.parser.ltlf import LTLfParser
from ltlf2dfa.ltlf2dfa import output2dot, MonaProgram

CDIR = os.getcwd()
MONA_PATH = os.path.join(CDIR, "refer_code", "mona.exe")

def formula_to_dfa(ifml, mona_name, dfa_name):
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
        print("[DFA Successful]", cmd)
        with open(opath, "r") as f:
            dfa_output = output2dot(f.read())
        
        return dfa_output
    return "! Not successful"

if __name__ == "__main__":
    print(formula_to_dfa("(a U b) & (c U d)", "and_until", "and_until"))
