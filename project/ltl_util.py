import numpy as np
from ltlf2dfa.parser.ltlf import LTLfParser
from ltlf2dfa.ltlf2dfa import to_dfa

parser = LTLfParser()
formula_str = "(a U b) & (c U d)"
formula = parser(formula_str)       # returns an LTLfFormula

to_dfa(formula)
# formula.to_dfa()


