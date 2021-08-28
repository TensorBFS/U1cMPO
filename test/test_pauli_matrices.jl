sigmaz = [1. 0. ; 0. -1.]
sigmap = [0. 1. ; 0.  0.]
sigmam = [0. 0. ; 1.  0.]
id = [1. 0. ; 0.  1.]
@test toarray(pauli(0)) == sigmaz
@test toarray(pauli(1)) == sigmap
@test toarray(pauli(-1)) == sigmam
@test toarray(pauli_id()) == id
