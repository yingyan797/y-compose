#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This file is part of ltlf2dfa.
#
# ltlf2dfa is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ltlf2dfa is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ltlf2dfa.  If not, see <https://www.gnu.org/licenses/>.
#

"""This module contains the implementation of Linear Temporal Logic on finite traces."""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ltlf_tools.base import (
    AtomicFormula,
    AtomSymbol,
    BinaryOperator,
    Formula,
    UnaryOperator,
)
from ltlf_tools.helpers import new_var
from ltlf_tools.ltlf2dfa import to_dfa
from ltlf_tools.pl import PLAtomic
from ltlf_tools.symbols import OpSymbol, Symbols

# Variable counter for generating unique variables
_var_counter = 0

def unique_var() -> str:
    """Generate a unique variable name."""
    global _var_counter
    _var_counter += 1
    return f"v_{_var_counter}"

def reset_var_counter() -> None:
    """Reset the variable counter for fresh variable names."""
    global _var_counter
    _var_counter = 0


class LTLfFormula(Formula, ABC):
    """A class for the LTLf formula."""

    def to_nnf(self) -> "LTLfFormula":
        """Convert an LTLf formula in NNF."""
        return self

    @abstractmethod
    def negate(self) -> "LTLfFormula":
        """Negate the formula."""

    def __repr__(self):
        """Get the representation."""
        return self.__str__()

    def to_mona(self, v: Optional[Any] = None, m: str = "max($)") -> str:
        """
        Tranform the LTLf formula into its encoding in MONA.

        :param v: a string for the variable name.
        :param m: a string for the max variable, representing the end of the trace.
        :return: a string.
        """

    def to_ldlf(self):
        """
        Tranform the formula into an equivalent LDLf formula.

        :return: an LDLf formula.
        """

    def to_dfa(self, mona_dfa_out: bool = False) -> str:
        """
        Translate into a DFA.

        :param mona_dfa_out: flag for DFA output in MONA syntax.
        """
        return to_dfa(self, mona_dfa_out)


class LTLfUnaryOperator(UnaryOperator[LTLfFormula], LTLfFormula, ABC):
    """A unary operator for LTLf."""


class LTLfBinaryOperator(BinaryOperator[LTLfFormula], LTLfFormula, ABC):
    """A binary operator for LTLf."""


class LTLfAtomic(AtomicFormula, LTLfFormula):
    """Class for LTLf atomic formulas."""

    name_regex = re.compile(r"[a-z][a-z0-9_]*")

    def negate(self):
        """Negate the formula."""
        return LTLfNot(self)

    def find_labels(self) -> List[AtomSymbol]:
        """Find the labels."""
        return PLAtomic(self.s).find_labels()

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf atomic formula."""
        if v != "0":
            return "({} in {})".format(v, self.s.upper())
        else:
            return PLAtomic(self.s).to_mona()

    # def to_ldlf(self):
    #     """Convert the formula to LDLf."""
    #     return LDLfDiamond(RegExpPropositional(PLAtomic(self.s)), LDLfLogicalTrue())


class LTLfTrue(LTLfAtomic):
    """Class for the LTLf True formula."""

    def __init__(self):
        """Initialize the formula."""
        super().__init__(Symbols.TRUE.value)

    def negate(self):
        """Negate the formula."""
        return LTLfFalse()

    def find_labels(self) -> List[AtomSymbol]:
        """Find the labels."""
        return list()

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding for False."""
        return Symbols.TRUE.value

    # def to_ldlf(self):
    #     """Convert the formula to LDLf."""
    #     return LDLfDiamond(RegExpPropositional(PLTrue()), LDLfLogicalTrue())


class LTLfFalse(LTLfAtomic):
    """Class for the LTLf False formula."""

    def __init__(self):
        """Initialize the formula."""
        super().__init__(Symbols.FALSE.value)

    def negate(self):
        """Negate the formula."""
        return LTLfTrue()

    def find_labels(self) -> List[AtomSymbol]:
        """Find the labels."""
        return list()

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding for False."""
        return Symbols.FALSE.value


class LTLfNot(LTLfUnaryOperator):
    """Class for the LTLf not formula."""

    @property
    def operator_symbol(self) -> OpSymbol:
        """Get the operator symbol."""
        return Symbols.NOT.value

    def to_nnf(self) -> LTLfFormula:
        """Transform to NNF."""
        if not isinstance(self.f, AtomicFormula):
            return self.f.negate().to_nnf()
        else:
            return self

    def negate(self) -> LTLfFormula:
        """Negate the formula."""
        return self.f

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf Not formula."""
        return "~({})".format(self.f.to_mona(v, m))

    # def to_ldlf(self):
    #     """Convert the formula to LDLf."""
    #     return LDLfNot(self.f.to_ldlf())


class LTLfAnd(LTLfBinaryOperator):
    """Class for the LTLf And formula."""

    @property
    def operator_symbol(self) -> OpSymbol:
        """Get the operator symbol."""
        return Symbols.AND.value

    def negate(self) -> LTLfFormula:
        """Negate the formula."""
        return LTLfOr([f.negate() for f in self.formulas])

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf And formula."""
        return "({})".format(" & ".join([f.to_mona(v, m) for f in self.formulas]))

    # def to_ldlf(self):
    #     """Convert the formula to LDLf."""
    #     return LDLfAnd([f.to_ldlf() for f in self.formulas])


class LTLfOr(LTLfBinaryOperator):
    """Class for the LTLf Or formula."""

    @property
    def operator_symbol(self) -> OpSymbol:
        """Get the operator symbol."""
        return Symbols.OR.value

    def negate(self) -> LTLfFormula:
        """Negate the formula."""
        return LTLfAnd([f.negate() for f in self.formulas])

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf Or formula."""
        return "({})".format(" | ".join([f.to_mona(v, m) for f in self.formulas]))

    # def to_ldlf(self):
    #     """Convert LTLf formula to LDLf."""
    #     return LDLfOr([f.to_ldlf() for f in self.formulas])


class LTLfImplies(LTLfBinaryOperator):
    """Class for the LTLf Implication formula."""

    @property
    def operator_symbol(self) -> OpSymbol:
        """Get the operator symbol."""
        return Symbols.IMPLIES.value

    def negate(self) -> LTLfFormula:
        """Negate the formula."""
        return self.to_nnf().negate()

    def to_nnf(self) -> LTLfFormula:
        """Transform to NNF."""
        first, second = self.formulas[0:2]
        final_formula = LTLfOr([LTLfNot(first).to_nnf(), second.to_nnf()])
        for subformula in self.formulas[2:]:
            final_formula = LTLfOr(
                [LTLfNot(final_formula).to_nnf(), subformula.to_nnf()]
            )
        return final_formula

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf Implication formula."""
        return self.to_nnf().to_mona(v, m)

    # def to_ldlf(self):
    #     """Convert the formula to LDLf."""
    #     f1 = (
    #         LTLfImplies(self.formulas[:-1]).to_ldlf()
    #         if len(self.formulas) > 2
    #         else self.formulas[0].to_ldlf()
    #     )
    #     f2 = self.formulas[-1].to_ldlf()
    #     return LDLfOr([LDLfNot(f1), f2])


class LTLfEquivalence(LTLfBinaryOperator):
    """Class for the LTLf Equivalente formula."""

    @property
    def operator_symbol(self) -> OpSymbol:
        """Get the operator symbol."""
        return Symbols.EQUIVALENCE.value

    def to_nnf(self) -> LTLfFormula:
        """Transform to NNF."""
        fs = self.formulas
        pos = LTLfAnd(fs)
        neg = LTLfAnd([LTLfNot(f) for f in fs])
        res = LTLfOr([pos, neg]).to_nnf()
        return res

    def negate(self) -> LTLfFormula:
        """Negate the formula."""
        return self.to_nnf().negate()

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf Equivalence formula."""
        return self.to_nnf().to_mona(v, m)

    # def to_ldlf(self):
    #     """Convert the formula to LDLf."""
    #     f1 = (
    #         LTLfImplies(self.formulas[:-1]).to_ldlf()
    #         if len(self.formulas) > 2
    #         else self.formulas[0].to_ldlf()
    #     )
    #     f2 = self.formulas[-1].to_ldlf()
    #     return LDLfAnd([LDLfOr([LDLfNot(f1), f2]), LDLfOr([f1, LDLfNot(f2)])])


class LTLfNext(LTLfUnaryOperator):
    """Class for the LTLf Next formula."""

    @property
    def operator_symbol(self) -> OpSymbol:
        """Get the operator symbol."""
        return Symbols.NEXT.value

    def to_nnf(self) -> LTLfFormula:
        """Transform to NNF."""
        return LTLfNext(self.f.to_nnf())

    def negate(self) -> LTLfFormula:
        """Negate the formula."""
        return LTLfWeakNext(self.f.negate())

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf Next formula."""
        ex_var = unique_var()
        if v != "0":
            return "(ex1 {0}: {0} in $ & {0}={1}+1 & {0}<={3} & {2})".format(
                ex_var, v, self.f.to_mona(ex_var, m), m
            )
        else:
            return "(ex1 {0}: {0} in $ & {0}=1 & {0}<={2} & {1})".format(
                ex_var, self.f.to_mona(ex_var, m), m
            )

    # def to_ldlf(self):
    #     """Convert the formula to LDLf."""
    #     return LDLfDiamond(
    #         RegExpPropositional(PLTrue()),
    #         LDLfAnd([self.f.to_ldlf(), LDLfNot(LDLfEnd())]),
    #     )


class LTLfWeakNext(LTLfUnaryOperator):
    """Class for the LTLf Weak Next formula."""

    @property
    def operator_symbol(self) -> OpSymbol:
        """Get the operator symbol."""
        return Symbols.WEAK_NEXT.value

    def to_nnf(self) -> LTLfFormula:
        """Transform to NNF."""
        return LTLfWeakNext(self.f.to_nnf())

    def negate(self) -> LTLfFormula:
        """Negate the formula."""
        return LTLfNext(self.f.negate())

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf WeakNext formula."""
        ex_var = unique_var()
        if v != "0":
            return "(({1} = {3}) | (ex1 {0}: {0} in $ & {0}={1}+1 & {0}<={3} & {2}))".format(
                ex_var, v, self.f.to_mona(ex_var, m), m
            )
        else:
            return "((0 = {2}) | (ex1 {0}: {0} in $ & {0}=1 & {0}<={2} & {1}))".format(
                ex_var, self.f.to_mona(ex_var, m), m
            )

    # def to_ldlf(self):
    #     """Convert the formula to LDLf."""
    #     return LDLfBox(
    #         RegExpPropositional(PLTrue()), LDLfOr([self.f.to_ldlf(), LDLfEnd()])
    #     )


class LTLfUntil(LTLfBinaryOperator):
    """Class for the LTLf Until formula."""

    @property
    def operator_symbol(self) -> OpSymbol:
        """Get the operator symbol."""
        return Symbols.UNTIL.value

    def to_nnf(self):
        """Transform to NNF."""
        return LTLfUntil([f.to_nnf() for f in self.formulas])

    def negate(self):
        """Negate the formula."""
        return LTLfRelease([f.negate() for f in self.formulas])

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf Until formula."""
        ex_var = unique_var()
        all_var = unique_var()
        f1 = self.formulas[0].to_mona(v=all_var, m=m)
        f2 = (
            LTLfUntil(self.formulas[1:]).to_mona(v=ex_var, m=m)
            if len(self.formulas) > 2
            else self.formulas[1].to_mona(v=ex_var, m=m)
        )
        return (
            "(ex1 {0}: {0} in $ & {1}<{0}&{0}<={5} & {2} & "
            "(all1 {3}: {3} in $ & {1}<={3}&{3}<{0} => {4}))".format(
                ex_var, v, f2, all_var, f1, m
            )
        )

    # def to_ldlf(self):
    #     """Convert the formula to LDLf."""
    #     f1 = self.formulas[0].to_ldlf()
    #     f2 = (
    #         LTLfUntil(self.formulas[1:]).to_ldlf()
    #         if len(self.formulas) > 2
    #         else self.formulas[1].to_ldlf()
    #     )
    #     return LDLfDiamond(
    #         RegExpStar(RegExpSequence([RegExpTest(f1), RegExpPropositional(PLTrue())])),
    #         LDLfAnd([f2, LDLfNot(LDLfEnd())]),
    #     )


class LTLfRelease(LTLfBinaryOperator):
    """Class for the LTLf Release formula."""

    @property
    def operator_symbol(self) -> OpSymbol:
        """Get the operator symbol."""
        return Symbols.RELEASE.value

    def to_nnf(self):
        """Transform to NNF."""
        return LTLfRelease([f.to_nnf() for f in self.formulas])

    def negate(self):
        """Negate the formula."""
        return LTLfUntil([f.negate() for f in self.formulas])

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf Release formula."""
        ex_var = unique_var()
        all_var = unique_var()
        f1 = self.formulas[0].to_mona(v=ex_var, m=m)
        f2 = (
            LTLfRelease(self.formulas[1:]).to_mona(v=all_var, m=m)
            if len(self.formulas) > 2
            else self.formulas[1].to_mona(v=all_var, m=m)
        )
        return (
            "((ex1 {0}: {0} in $ & {1}<={0}&{0}<={5} & {2} & "
            "(all1 {3}: {3} in $ & {1}<={3}&{3}<={0} => {4})) | (all1 {3}: "
            "{3} in $ & {1}<={3}&{3}<={5} => {4}))".format(
                ex_var, v, f1, all_var, f2, m
            )
        )

    # def to_ldlf(self):
    #     """Convert the formula to LDLf."""
    #     f1 = self.formulas[0].to_ldlf()
    #     f2 = (
    #         LTLfRelease(self.formulas[1:]).to_ldlf()
    #         if len(self.formulas) > 2
    #         else self.formulas[1].to_ldlf()
    #     )
    #     return LDLfBox(
    #         RegExpStar(
    #             RegExpSequence([RegExpTest(LDLfNot(f1)), RegExpPropositional(PLTrue())])
    #         ),
    #         LDLfOr([f2, LDLfEnd()]),
    #     )


class LTLfEventually(LTLfUnaryOperator):
    """Class for the LTLf Eventually formula."""

    @property
    def operator_symbol(self) -> OpSymbol:
        """Get the operator symbol."""
        return Symbols.EVENTUALLY.value

    def to_nnf(self) -> LTLfFormula:
        """Transform to NNF."""
        return LTLfUntil([LTLfTrue(), self.f])

    def negate(self) -> LTLfFormula:
        """Negate the formula."""
        return self.to_nnf().negate()

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf Eventually formula."""
        return LTLfUntil([LTLfTrue(), self.f]).to_mona(v, m)

    # def to_ldlf(self):
    #     """Convert the formula to LDLf."""
    #     return LDLfDiamond(
    #         RegExpStar(RegExpPropositional(PLTrue())),
    #         LDLfAnd([self.f.to_ldlf(), LDLfNot(LDLfEnd())]),
    #     )


class LTLfAlways(LTLfUnaryOperator):
    """Class for the LTLf Always formula."""

    @property
    def operator_symbol(self) -> OpSymbol:
        """Get the operator symbol."""
        return Symbols.ALWAYS.value

    def to_nnf(self) -> LTLfFormula:
        """Transform to NNF."""
        return LTLfRelease([LTLfFalse(), self.f.to_nnf()])

    def negate(self) -> LTLfFormula:
        """Negate the formula."""
        return self.to_nnf().negate()

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf Always formula."""
        return LTLfRelease([LTLfFalse(), self.f]).to_mona(v, m)

    # def to_ldlf(self):
    #     """Convert the formula to LDLf."""
    #     return LDLfBox(
    #         RegExpStar(RegExpPropositional(PLTrue())),
    #         LDLfOr([self.f.to_ldlf(), LDLfEnd()]),
    #     )


class LTLfLast(LTLfFormula):
    """Class for the LTLf Last formula."""

    def to_nnf(self) -> LTLfFormula:
        """Transform to NNF."""
        return LTLfAnd([LTLfWeakNext(LTLfFalse()), LTLfNot(LTLfEnd())]).to_nnf()

    def negate(self) -> LTLfFormula:
        """Negate the formula."""
        return self.to_nnf().negate()

    def find_labels(self) -> List[AtomSymbol]:
        """Find the labels."""
        return list()

    def _members(self):
        return (Symbols.LAST.value,)

    def __str__(self):
        """Get the string representation."""
        return Symbols.LAST.value

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf Last formula."""
        return LTLfWeakNext(LTLfFalse()).to_mona(v, m)

    # def to_ldlf(self):
    #     """Convert the formula to LDLf."""
    #     return LDLfDiamond(RegExpPropositional(PLTrue()), LDLfEnd())


class LTLfEnd(LTLfFormula):
    """Class for the LTLf End formula."""

    def find_labels(self) -> List[AtomSymbol]:
        """Find the labels."""
        return list()

    def _members(self):
        return (Symbols.END.value,)

    def to_nnf(self) -> LTLfFormula:
        """Transform to NNF."""
        return LTLfAlways(LTLfFalse()).to_nnf()

    def negate(self) -> LTLfFormula:
        """Negate the formula."""
        return self.to_nnf().negate()

    def __str__(self):
        """Get the string representation."""
        return "_".join(map(str, self._members()))

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf End formula."""
        return "({0} = {1})".format(v, m)


class LTLfThen(LTLfBinaryOperator):
    """Class for the LTLf Then formula.
    
    The formula f1 Then f2 means that f1 holds up to a breaking point,
    and f2 holds from the breaking point onwards.
    """

    @property
    def operator_symbol(self) -> OpSymbol:
        """Get the operator symbol."""
        return Symbols.THEN.value  # Using "Then" as the symbol

    def to_nnf(self) -> LTLfFormula:
        """Transform to NNF."""
        return LTLfThen([f.to_nnf() for f in self.formulas])

    def negate(self) -> LTLfFormula:
        """Negate the formula."""
        # The negation of (f1 Then f2) is (¬f1 Then ¬f2)
        return LTLfThen([f.negate() for f in self.formulas])

    def to_mona(self, v="0", m="max($)") -> str:
        """Return the MONA encoding of an LTLf Then formula."""
        # We need a break point variable
        break_var = unique_var()
        ex_var = unique_var()
        
        # Formula 1 applies up to the break point
        f1 = self.formulas[0].to_mona(v=ex_var, m=break_var)
        
        # Formula 2 applies from the break point to the end
        if len(self.formulas) > 2:
            f2 = LTLfThen(self.formulas[1:]).to_mona(v=break_var, m=m)
        else:
            f2 = self.formulas[1].to_mona(v=break_var, m=m)
        
        # The variable used in Formula 2 to check positions
        return (
            "(ex1 {0}: {0} in $ & {1}<{0}&{0}<={5} & {2} & "
            "ex1 {3}: {3} in $ & {1}<={3}&{3}<{0} & {4})".format(
                break_var, v, f2, ex_var, f1, m
            )
        )

    # def to_ldlf(self):
    #     """Convert the formula to LDLf."""
    #     # Implementation would go here
    #     pass