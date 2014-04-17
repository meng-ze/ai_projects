from logic import *
from wumpus_environment import *
from wumpus_kb import *
from wumpus_planners import *
import minisat as msat
from time import clock
import sys


#-------------------------------------------------------------------------------

def minisat(clauses, query = None, variable = None, value = True, verbose = False):
    """ Interface to minisat
    <query> is simply added as to the list of <clauses>
    
    Set <variable> to a particular <value> in order to test SAT
    assuming any instance of that variable has that value.
    
    Otherwise, with defaults, will perform normal SAT on <clauses>+<query>
    """
    c = None
    if verbose:
        print 'minisat({0}):'.format(query),
    if not query:
        c = clauses
    else:
        c = clauses + [query]
    m = msat.Minisat()
    s = m.solve(clauses, variable, value)
    print "clauses: " , s.success
    s = m.solve(c, variable, value)
    if verbose:
        print s.success
    return s

#-------------------------------------------------------------------------------

class PropKB_SAT(PropKB):

    def tell(self, sentence):
        if sentence: super(PropKB_SAT,self).tell(sentence)

    def load_sentences(self, sentences):
        for sentence in sentences: self.tell(sentence)

    def ask(self, query):
        """ Assumes query is a single positive proposition """
        if isinstance(query,str):
            query = expr(query)
        sT = minisat(self.clauses, None, variable=query, value=True, verbose=False)
        sF = minisat(self.clauses, None, variable=query, value=False, verbose=False)
        if sT.success == sF.success:
            return None
        else:
            return sT.success

kb = PropKB_SAT()
for line in sys.stdin:
    print line