import Base: *

@doc raw""" 
cMPO (for now, only nearest-neighboring (NN) interactions) 
```math
\left( \begin{array}{cc}
  Q & \mathbf{R}\\
  \mathbf{L} & 0
\end{array} \right)
```
"""
struct cmpo{Ti<:Integer, Tf<:AbstractFloat}
    """ Q matrix"""
    Q::u1_matrix{Ti, Tf}
    """ L tensor """
    L::Array{u1_matrix{Ti, Tf}, 1}
    """ R tensor """
    R::Array{u1_matrix{Ti, Tf}, 1}
    #P::Array{u1_matrix{Ti, Tf}, 2}
end

""" cMPS """
struct cmps{Ti<:Integer, Tf<:AbstractFloat}
    """ Q matrix """
    Q::u1_matrix{Ti, Tf}
    """ R tensor """
    R::Array{u1_matrix{Ti, Tf}, 1}
end

"""
    eye_u1m(op)

Given a cMPO, generate an identity matrix according to its U(1) quantum number sectors.
"""
function eye_u1m(op::cmpo{Ti, Tf}) where {Ti, Tf}
    eye_u1m(Tf, op.Q.d_vec)
end
"""
    eye_u1m(psi)

Given a cMPS, generate an identity matrix according to its U(1) quantum number sectors.
"""
function eye_u1m(psi::cmps{Ti, Tf}) where {Ti, Tf}
    eye_u1m(Tf, psi.Q.d_vec)
end
"""
    reflect(psi)

reflect the matrices contained in the input cMPS `psi`.
See also: [`reflect(m)`](@ref)
"""
function reflect(psi::cmps)
    cmps(reflect(psi.Q), reflect.(psi.R))
end

"""
    *(op, psi)

Act the cMPO `op` onto the cMPS `psi`, and return the new cMPS.
"""
function *(op::cmpo, psi::cmps)
    op_id = eye_u1m(op)
    psi_id = eye_u1m(psi)
    Q = op.Q ⊗ psi_id + op_id ⊗ psi.Q + sum(op.R .⊗ psi.R)
    R = op.L .⊗ [psi_id]
    cmps(Q, R)
end

"""
    *(phi, psi)

Return the K-matrix from the input cMPS `phi` and `psi`.
"""
function *(phi::cmps, psi::cmps)
    phi_id = eye_u1m(phi)
    psi_id = eye_u1m(psi)

    Q = phi.Q ⊗ psi_id + phi_id ⊗ psi.Q + sum(phi.R .⊗ psi.R)
    Q
end

"""
    *(phi, op)

Act the cMPO `op` to the left cMPS `phi`.
"""
function *(phi::cmps, op::cmpo)
    op_id = eye_u1m(op)
    phi_id = eye_u1m(phi)

    Q = phi_id ⊗ op.Q + phi.Q ⊗ op_id + sum(phi.R .⊗ op.L)
    R = [phi_id] .⊗ op.R
    cmps(Q, R)
end

"""
    *(op1, op2)

Multiply cMPO `op1` with cMPO `op2`. 
"""
function *(op1::cmpo, op2::cmpo)
    op1_id = eye_u1m(op1)
    op2_id = eye_u1m(op2)

    Q = op1_id ⊗ op2.Q + op1.Q ⊗ op2_id + sum(op1.R .⊗ op2.L)
    L = op1.L .⊗ [op2_id]
    R = [op1_id] .⊗ op2.R

    cmpo(Q, L, R)
end

"""
    ovlp(phi, psi, beta)

Given inversed temperature `beta`, calculate the log of the overlap between the cMPSs `phi` and `psi`
"""
function ovlp(phi::cmps, psi::cmps, beta::AbstractFloat)
    rho = phi * psi
    log_tr_expm(rho, beta)
end

"""
    diagQ(psi)

Perform gauge transformation to cMPS `psi` so that its Q matrix is diagonalized.
"""
function diagQ(psi::cmps)
    w, v = eigh(psi.Q)

    Q = init_diag_u1m(psi.Q.d_vec, w)
    R = [transpose(v)] .* psi.R .* [v]
    cmps(Q, R)
end
