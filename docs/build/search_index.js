var documenterSearchIndex = {"docs":
[{"location":"index.html","page":"Home","title":"Home","text":"CurrentModule = U1cMPO","category":"page"},{"location":"index.html#U1cMPO","page":"Home","title":"U1cMPO","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Modules = [U1cMPO]","category":"page"},{"location":"index.html#U1cMPO.cmpo","page":"Home","title":"U1cMPO.cmpo","text":"cMPO (for now, only nearest-neighboring (NN) interactions) \n\nleft( beginarraycc\n  Q  mathbfR\n  mathbfL  0\nendarray right)\n\n\n\n\n\n","category":"type"},{"location":"index.html#U1cMPO.cmps","page":"Home","title":"U1cMPO.cmps","text":"cMPS \n\n\n\n\n\n","category":"type"},{"location":"index.html#U1cMPO.u1_matrix","page":"Home","title":"U1cMPO.u1_matrix","text":"definition for matrices with U(1) block structures. \n\n\n\n\n\n","category":"type"},{"location":"index.html#Base.:*-Tuple{Real, u1_matrix}","page":"Home","title":"Base.:*","text":"*(x, m)\n\nNumber-Matrix multiplication.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.:*-Tuple{cmpo, cmpo}","page":"Home","title":"Base.:*","text":"*(op1, op2)\n\nMultiply cMPO op1 with cMPO op2. \n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.:*-Tuple{cmpo, cmps}","page":"Home","title":"Base.:*","text":"*(op, psi)\n\nAct the cMPO op onto the cMPS psi, and return the new cMPS.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.:*-Tuple{cmps, cmpo}","page":"Home","title":"Base.:*","text":"*(phi, op)\n\nAct the cMPO op to the left cMPS phi.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.:*-Tuple{cmps, cmps}","page":"Home","title":"Base.:*","text":"*(phi, psi)\n\nReturn the K-matrix from the input cMPS phi and psi.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.:*-Tuple{u1_matrix, Real}","page":"Home","title":"Base.:*","text":"*(m, x)\n\nNumber-Matrix multiplication.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.:*-Union{Tuple{Tf}, Tuple{Ti}, Tuple{u1_matrix{Ti, Tf}, u1_matrix{Ti, Tf}}} where {Ti, Tf}","page":"Home","title":"Base.:*","text":"*(m1, m2)\n\nMatrix multiplication.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.:+-Tuple{u1_matrix, u1_matrix}","page":"Home","title":"Base.:+","text":"+(m1, m2)\n\nAdd two matrices.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.:--Tuple{u1_matrix, u1_matrix}","page":"Home","title":"Base.:-","text":"-(m1, m2)\n\nCalculate m1 - m2.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.:==-Tuple{u1_matrix, u1_matrix}","page":"Home","title":"Base.:==","text":"==(m1, m2)\n\nReturn whether two matrices are identical.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.adjoint-Tuple{u1_matrix}","page":"Home","title":"Base.adjoint","text":"adjoint(m)\n\nThis function does nothing to the matrix.  It is useful when calculating the adjoint (transpose) of a cMPO.  \n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.isapprox-Tuple{u1_matrix, u1_matrix}","page":"Home","title":"Base.isapprox","text":"isapprox(m1, m2)\n\nReturn whether two matrices are approximately identical.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.kron-Tuple{u1_matrix, u1_matrix}","page":"Home","title":"Base.kron","text":"kron(m1, m2)\n\nTensor product of matrices.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.transpose-Union{Tuple{u1_matrix{Ti, Tf}}, Tuple{Tf}, Tuple{Ti}} where {Ti, Tf}","page":"Home","title":"Base.transpose","text":"transpose(m)\n\nTake the transpose of a u1_matrix.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Base.vec-Union{Tuple{u1_matrix{Ti, Tf}}, Tuple{Tf}, Tuple{Ti}, Tuple{u1_matrix{Ti, Tf}, Bool}} where {Ti, Tf}","page":"Home","title":"Base.vec","text":"vec(u1_matrix, is_diag=false)\n\nConvert all the parameters (the data within the blocks) to a vector.  If is_diag is set to true, then only take the diagonal elements.\n\n\n\n\n\n","category":"method"},{"location":"index.html#ChainRulesCore.rrule-Tuple{typeof(init_u1m_by_params), Vector{var\"#s5\"} where var\"#s5\"<:Integer, Integer, Vector{var\"#s4\"} where var\"#s4\"<:AbstractFloat}","page":"Home","title":"ChainRulesCore.rrule","text":"rules for U1Matrix\n\n\n\n\n\n","category":"method"},{"location":"index.html#ChainRulesCore.rrule-Tuple{typeof(kron), AbstractMatrix{T} where T, AbstractMatrix{T} where T}","page":"Home","title":"ChainRulesCore.rrule","text":"rules missed in LinearAlgebra\n\n\n\n\n\n","category":"method"},{"location":"index.html#LinearAlgebra.tr-Tuple{u1_matrix}","page":"Home","title":"LinearAlgebra.tr","text":"tr(m)\n\nTrace of matrix.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.:⊗-Tuple{u1_matrix, u1_matrix}","page":"Home","title":"U1cMPO.:⊗","text":"⊗(m1, m2)\n\nTensor product of matrices.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.C_matrix-Union{Tuple{Tf}, Tuple{Ti}, Tuple{cmps{Ti, Tf}, AbstractFloat, AbstractFloat}} where {Ti, Tf}","page":"Home","title":"U1cMPO.C_matrix","text":"C_matrix(psi, beta, tau)\n\nA util function for the the calculation of the von-Neumann entropy which calculates the coefficent matrix.  psi is cMPS, beta is the inversed temperature, and tau is the interval in the imaginary time. We have inserted some unitary matrices to maintain the U(1) block structure of the matrix. \n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.L2_theta_0-Tuple{Integer}","page":"Home","title":"U1cMPO.L2_theta_0","text":"L2_theta_0(lmax)\n\nReturn the matrix representation for the kinetic term in quantum rotor model (no theta term).\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.L2_theta_pi-Tuple{Integer}","page":"Home","title":"U1cMPO.L2_theta_pi","text":"L2_theta_0(lmax)\n\nReturn the matrix representation for the kinetic term in quantum rotor model (theta=pi).\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.X_theta_0-Tuple{Integer, Integer}","page":"Home","title":"U1cMPO.X_theta_0","text":"X_theta_0(M, lmax)\n\nReturn matrix representation of spherical tensor operator for ordinary quantum rotors.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.X_theta_pi-Tuple{Integer, Integer}","page":"Home","title":"U1cMPO.X_theta_pi","text":"X_theta_pi(M, lmax)\n\nReturn matrix representation of spherical tensor operator for quantum rotors  decorated with a magnetic monopole (theta=pi).\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.add_to_submat!-Tuple{u1_matrix, Matrix{var\"#s4\"} where var\"#s4\"<:AbstractFloat, Integer, UnitRange{var\"#s3\"} where var\"#s3\"<:Integer, UnitRange{var\"#s1\"} where var\"#s1\"<:Integer}","page":"Home","title":"U1cMPO.add_to_submat!","text":"add_to_submat!(m, subm_to_add, l_label, l_indices, r_indices)\n\nInplace modification of the U1 matrix m by adding some small matrix subm_to_add to it. The small matrix is added to the block corresponding to the quantum number l_label,  within the sub-block indicated by the index ranges l_indices and r_indices.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.add_to_submat!-Tuple{u1_matrix, Matrix{var\"#s6\"} where var\"#s6\"<:AbstractFloat, Integer}","page":"Home","title":"U1cMPO.add_to_submat!","text":"add_to_submat!(m, subm_to_add, l_label)\n\nInplace modification of the U1 matrix m by adding some small matrix subm_to_add to it. The small matrix is added to the whole block corresponding to the quantum number l_label. \n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.blk_chk-Tuple{u1_matrix, u1_matrix}","page":"Home","title":"U1cMPO.blk_chk","text":"blk_chk(m1, m2)\n\nReturn whether two matrices has the same block structure.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.d_vec_0-Tuple{Integer}","page":"Home","title":"U1cMPO.d_vec_0","text":"d_vec_0(lmax)\n\nReturn the U(1) block structure given the largest l kept in the case of no theta term.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.d_vec_pi-Tuple{Integer}","page":"Home","title":"U1cMPO.d_vec_pi","text":"d_vec_pi(lmax)\n\nReturn the U(1) block structure given the largest l kept in the case of theta=pi.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.diagQ-Tuple{cmps}","page":"Home","title":"U1cMPO.diagQ","text":"diagQ(psi)\n\nPerform gauge transformation to cMPS psi so that its Q matrix is diagonalized.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.eigh-Union{Tuple{u1_matrix{Ti, Tf}}, Tuple{Tf}, Tuple{Ti}} where {Ti, Tf}","page":"Home","title":"U1cMPO.eigh","text":"eigh(m)\n\nEigendecomposition of a (symmetric) U1 matrix.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.energy-Tuple{cmpo, cmps, AbstractFloat}","page":"Home","title":"U1cMPO.energy","text":"energy(T, psi, beta)\n\nCalculate energy from cMPO T, cMPS psi at inversed temperature beta.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.eye_u1m-Tuple{Type{var\"#s5\"} where var\"#s5\"<:AbstractFloat, Vector{var\"#s4\"} where var\"#s4\"<:Integer}","page":"Home","title":"U1cMPO.eye_u1m","text":"eye_u1m(T, d_vec)\n\nInitialize an identity matrix of datatype T, and assign a block structure to it.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.eye_u1m-Union{Tuple{cmpo{Ti, Tf}}, Tuple{Tf}, Tuple{Ti}} where {Ti, Tf}","page":"Home","title":"U1cMPO.eye_u1m","text":"eye_u1m(op)\n\nGiven a cMPO, generate an identity matrix according to its U(1) quantum number sectors.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.eye_u1m-Union{Tuple{cmps{Ti, Tf}}, Tuple{Tf}, Tuple{Ti}} where {Ti, Tf}","page":"Home","title":"U1cMPO.eye_u1m","text":"eye_u1m(psi)\n\nGiven a cMPS, generate an identity matrix according to its U(1) quantum number sectors.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.f_and_gf_gen-Tuple{cmpo, Function, AbstractFloat}","page":"Home","title":"U1cMPO.f_and_gf_gen","text":"f_and_gf_gen(T, params_to_cmps, beta)\n\nGiven a function params_to_cmps that convert parameter vector to cmps, the Hermtian cMPO T, and the inversed temperature beta,  generate functions that calculate free energy and the corresponding gradient from parameters.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.fidel_and_gfidel_gen-Tuple{cmps, Function, AbstractFloat}","page":"Home","title":"U1cMPO.fidel_and_gfidel_gen","text":"fidel_and_gfidel_gen(phi, params_to_cmps, beta)\n\nGiven a function params_to_cmps that convert parameter vector to cmps, the target cMPS phi, and the inversed temperature beta,  generate functions that calculate free energy and the corresponding gradient from parameters.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.free_energy-Tuple{cmpo, cmps, AbstractFloat}","page":"Home","title":"U1cMPO.free_energy","text":"free_energy(T, psi, beta)\n\ngiven cMPS psi, Hermitian cMPO T and the inversed temperature beta,  calculate the free energy. \n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.init_diag_u1m-Union{Tuple{Tf}, Tuple{Ti}, Tuple{Vector{Ti}, Vector{Tf}}} where {Ti<:Integer, Tf<:AbstractFloat}","page":"Home","title":"U1cMPO.init_diag_u1m","text":"init_diag_u1m(d_vec, params)\n\nInitialize a diagonal matrix from a vector params, and then assign a block structure to it.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.init_u1m_by_params-Tuple{Vector{var\"#s6\"} where var\"#s6\"<:Integer, Integer, Vector{var\"#s7\"} where var\"#s7\"<:AbstractFloat}","page":"Home","title":"U1cMPO.init_u1m_by_params","text":"init_u1m_by_params(d_vec, qn, params)\n\nAn alternative initialization for u1_matrix from a vector of parameters.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.isometry_truncate-Union{Tuple{Tf}, Tuple{Ti}, Tuple{u1_matrix{Ti, Tf}, u1_matrix{Ti, Tf}, Vector{BitVector}}} where {Ti, Tf}","page":"Home","title":"U1cMPO.isometry_truncate","text":"isometry_truncate(m, v, msk_vec)\n\nTruncate the U1 matrix m by an isometry matrix.  We implement this by first transform the matrix by a orthogonal matrix v  that has the same block structure with m, and then truncate the transformed U1 matrix  according to the mask msk_vec, block by block. \n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.klein-Tuple{cmps, cmps, AbstractFloat}","page":"Home","title":"U1cMPO.klein","text":"klein(Lpsi, psi, beta)\n\nCalculate the klein bottle entropy from left cMPS Lpsi and right cMPS psi at  inversed temperature beta. \n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.kron_bookkeeping-Tuple{Vector{var\"#s4\"} where var\"#s4\"<:Integer, Vector{var\"#s3\"} where var\"#s3\"<:Integer}","page":"Home","title":"U1cMPO.kron_bookkeeping","text":"kron_bookkeeping(d_vec1, d_vec2)\n\nUtilization function for the calculation of the tensor product.  Given two U(1) quantum number sectors, return \n\nthe new quantum number sector for the tensor product matrix. \na function that maps the qunatum numbers (n1, n2) corresponding to input quantum number sectors  to the sub-block location (l, x0, x1) within the new matrix, where l is the new quantum number and x0 and x1 represent the beginning and ending point within in the block.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.l_labels_f-Tuple{Integer, Integer}","page":"Home","title":"U1cMPO.l_labels_f","text":"l_labels_f(nlen, qn)\n\nGiven the total number nlen of the U(1) quantum numbers and the quantum number qn of the U1 matrix,  return a vector that maps the index of the block to the U(1) quantum number of the block.  For example, when qn=1, the first block corresponds to quantum number 1. when qn=-1, however, the first block corresponds to quantum number 2.  (The U(1) quantum number is assumed to start from 1 to nlen.) \n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.log_tr_expm-Tuple{u1_matrix, AbstractFloat}","page":"Home","title":"U1cMPO.log_tr_expm","text":"log_tr_expm(m, beta)\n\nCalculates \n\nlog mathrmTr exp(beta M)\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.logsumexp-Tuple{Vector{var\"#s5\"} where var\"#s5\"<:AbstractFloat}","page":"Home","title":"U1cMPO.logsumexp","text":"logsumexp(w)\n\nCalculation of logsumexp.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.ovlp-Tuple{cmps, cmps, AbstractFloat}","page":"Home","title":"U1cMPO.ovlp","text":"ovlp(phi, psi, beta)\n\nGiven inversed temperature beta, calculate the log of the overlap between the cMPSs phi and psi\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.params_len-Tuple{Vector{var\"#s6\"} where var\"#s6\"<:Integer, Integer}","page":"Home","title":"U1cMPO.params_len","text":"params_len(d_vec, qn)\n\nGiven the quantum number sector d_vec and the quantum number of the matrix qn, determine the total number of the parameters.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.pauli-Tuple{Integer}","page":"Home","title":"U1cMPO.pauli","text":"pauli(M)\n\nGenerate Pauli matrix (σ0, σ+, and σ-).\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.pauli_id-Tuple{}","page":"Home","title":"U1cMPO.pauli_id","text":"pauli_id()\n\nGenerate 2×2 identity matrix.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.reduced_density_matrix-Union{Tuple{Tf}, Tuple{Ti}, Tuple{cmps{Ti, Tf}, AbstractFloat}} where {Ti, Tf}","page":"Home","title":"U1cMPO.reduced_density_matrix","text":"reduced_density_matrix(psi, beta)\n\nFor cMPS psi with imaginary-time length beta, construct the reduced density matrix. We have inserted some unitaries to maintain the U(1) block structure of the matrix.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.reflect-Tuple{cmps}","page":"Home","title":"U1cMPO.reflect","text":"reflect(psi)\n\nreflect the matrices contained in the input cMPS psi. See also: reflect(m)\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.reflect-Tuple{u1_matrix}","page":"Home","title":"U1cMPO.reflect","text":"reflect(m)\n\nreflect a u1_matrix. Equivalent to the following unitary tranformation \n\nleft( beginarraycccc\n       1\n     1  \n   udots    \n  1      \nendarray right)\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.submat-Tuple{u1_matrix, Integer}","page":"Home","title":"U1cMPO.submat","text":"submat(m, n)\n\nReturn the block corresponding to U(1) quantum number n. \n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.symmetrize-Tuple{u1_matrix}","page":"Home","title":"U1cMPO.symmetrize","text":"symmetrize(m)\n\nSymmetrize the matrix by \n\n(M + M^T)  2\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.toarray-Union{Tuple{u1_matrix{Ti, Tf}}, Tuple{Tf}, Tuple{Ti}} where {Ti, Tf}","page":"Home","title":"U1cMPO.toarray","text":"to_array(m)\n\nRepresent the U1 matrix as an ordinary 2D-array.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.trunc_msk_gen-Tuple{Vector{var\"#s4\"} where var\"#s4\"<:Integer, Vector{var\"#s3\"} where var\"#s3\"<:AbstractFloat, Integer}","page":"Home","title":"U1cMPO.trunc_msk_gen","text":"trunc_msk_gen(d_vec, w, num_kept)\n\nGenerate msk_vec for the function isometry_truncate(m, v, msk_vec). According to the block structure d_vec and some spectrum w, keep num_kept of states in the spectrum. Return a vector of BitVector that tells how to truncate a U1 matrix, block by block.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.truncate_according_to_rdm-Tuple{cmps, Integer, AbstractFloat}","page":"Home","title":"U1cMPO.truncate_according_to_rdm","text":"truncate_according_to_rdm(psi, chi, beta)\n\nTruncate cMPS psi to target bond dimension chi at inversed temperature beta  according to the reduced density matrix eigenvalues.  The new U(1) quantum number sectors can be automatically determined from this process.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.von_neumann_entropy-Tuple{cmps, AbstractFloat, AbstractFloat}","page":"Home","title":"U1cMPO.von_neumann_entropy","text":"von_neumann_entropy(psi, beta, tau)\n\nFor cMPS psi, calculate the von-Neumann entropy between intervals τ and β-τ.\n\n\n\n\n\n","category":"method"},{"location":"index.html#U1cMPO.zero_u1m-Tuple{Type{var\"#s5\"} where var\"#s5\"<:AbstractFloat, Vector{var\"#s4\"} where var\"#s4\"<:Integer, Integer}","page":"Home","title":"U1cMPO.zero_u1m","text":"zero_u1m(T, d_vec)\n\nInitialize a zero matrix of datatype T, and assign a block structure to it.\n\n\n\n\n\n","category":"method"}]
}
