from typing import Callable, Sequence, Mapping, Tuple
from utils.generic_typevars import S, A

"""
VFType (= Callable[[S], float]) is the type that represents a value
function for the most general situation. Instead of thinking of a value
function as a dictionary from states to returns, think of a value function
as a function from states to returns. This representation works
for all forms of MDPs, discrete/finite or continuous state spaces. 

QFType (= Callable[[S], Callable[[A], float]]) is the type that represents
the action value function (or Q function) for the most general situation.
Instead of thinking of a Q Function as a dictionary from state, action pairs
to returns, think of a Q Function as a function from states to {functions
from actions to returns}. This representation works for all forms of MDPs,
discrete/finite or continuous state spaces and action spaces.

PolicyType (= Callable[[S], Callable[[int], Sequence[A]]]) is the type
thet represents a stochastic policy for the most general situation. Instead 
of thinking of a policy as a dictionary from states to {dictionary from
actions to probabilities}, think of a policy as a function from states to
probability distributions where a probability distribution has the most
general representation (that would work for discrete or finite action spaces).
This general representation of a probability distribution is a function 
that takes as input the number of action samples and produces as output
a sequence of actions drawn from that probability distribution. In other
words, we can make the probability distribution as fine or coarse as we
wnt by controlling the input to this function (the requested number of
sample points).

VFDictType (= Mapping[S, float]) is the type that represents a value function
for a finite set of states amd hence, is represented as a data structure rather
than a function. One can always produce a VFType from a VFDictType by wrapping
the dictionary with a function.

QFDictType (= Mapping[S, Mapping[A, float]) is the type that represents an
action value function (or Q function) for a finite set of states and actions.
Hence, it is represented as a data structure rather than as a function.
One can always produce a QFType from a QFDictType by wrapping the dictionary
of dictionaries with a function returning a function.

PolicyActDictType (= Callable[[S], Mapping[A, float]]) is the type that 
represents a policy for arbitrary state space and finite action spaces.

The S*f types are types required for tabular methods which work with
nested dictionaries (rather than functions)
"""

VFType = Callable[[S], float]
QFType = Callable[[S], Callable[[A], float]]
PolicyType = Callable[[S], Callable[[int], Sequence[A]]]

VFDictType = Mapping[S, float]
QFDictType = Mapping[S, Mapping[A, float]]
PolicyActDictType = Callable[[S], Mapping[A, float]]

SSf = Mapping[S, Mapping[S, float]]
SSTff = Mapping[S, Mapping[S, Tuple[float, float]]]
STSff = Mapping[S, Tuple[Mapping[S, float], float]],
SAf = Mapping[S, Mapping[A, float]]
SASf = Mapping[S, Mapping[A, Mapping[S, float]]]
SASTff = Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]
SATSff = Mapping[S, Mapping[A, Tuple[Mapping[S, float], float]]]

