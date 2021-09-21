"""
    U1Matrix.jl

Definitions and methods for matrices with U(1) block structures.
todo: add abstract type AbstractU1Matrix to make long-range P block work
"""

import Base: ==, +, -, *, isapprox, transpose, adjoint, zeros, kron, vec

using LinearAlgebra
import LinearAlgebra: tr

""" definition for matrices with U(1) block structures. """
struct u1_matrix{Ti<:Integer, Tf<:AbstractFloat}
    """ The U(1) quantum number sectors. """
    d_vec::Array{Ti, 1}
    """ 
    The quantum number of this matrix. 
    - qn=0: block diagonalized.
    - qn>0: act as raising operators.
    - qn<0: act as lowering operators.
    """
    qn::Ti
    """ The data contained in the matrix. """
    submat_arr::Array{Array{Tf, 2}, 1}
end

"""
    init_u1m_by_params(d_vec, qn, params)

An alternative initialization for u1_matrix from a vector of parameters.
"""
function init_u1m_by_params(d_vec::Vector{<:Integer}, qn::Integer, params::Vector{<:AbstractFloat})
    nlen = length(d_vec)

    # left and right labels, used to determine block shape
    l_labels = l_labels_f(nlen, qn)
    r_labels = l_labels .+ qn
    d_vec_l = d_vec[l_labels]
    d_vec_r = d_vec[r_labels]

    submat_arr = Array{Array{typeof(params[1]), 2}, 1}([])
    ix = 1
    for (dl, dr) in zip(d_vec_l, d_vec_r)
        push!(submat_arr, reshape(params[ix:ix+dl*dr-1], (dl, dr)) )
        ix +=  dl*dr
    end

    u1_matrix(d_vec, qn, submat_arr)
end

"""
    init_diag_u1m(d_vec, params)

Initialize a diagonal matrix from a vector `params`, and then assign a block structure to it.
"""
function init_diag_u1m(d_vec::Array{Ti, 1}, params::Array{Tf, 1}) where {Ti<:Integer, Tf<:AbstractFloat}
    submat_arr = Array{Array{Tf, 2}, 1}([])
    ix = 1
    for dl in d_vec
        push!(submat_arr, Diagonal(params[ix:ix+dl-1]) )
        ix += dl
    end
    u1_matrix(d_vec, 0, submat_arr)
end

"""
    eye_u1m(T, d_vec)

Initialize an identity matrix of datatype `T`, and assign a block structure to it.
"""
function eye_u1m(T::Type{<:AbstractFloat}, d_vec::Array{<:Integer, 1})
    plen = sum(d_vec)
    params = ones(T, plen)
    init_diag_u1m(d_vec, params)
end

"""
    zero_u1m(T, d_vec)

Initialize a zero matrix of datatype `T`, and assign a block structure to it.
"""
function zero_u1m(T::Type{<:AbstractFloat}, d_vec::Array{<:Integer, 1}, qn::Integer)
    plen = params_len(d_vec, qn)
    params = zeros(T, plen)
    init_u1m_by_params(d_vec, qn, params)
end

"""
    blk_chk(m1, m2)

Return whether two matrices has the same block structure.
"""
function blk_chk(m1::u1_matrix, m2::u1_matrix)
    (m1.d_vec == m2.d_vec) && (m1.qn == m2.qn)
end

"""
    ==(m1, m2)

Return whether two matrices are identical.
"""
function ==(m1::u1_matrix, m2::u1_matrix)
    blk_chk(m1, m2) && (m1.submat_arr == m2.submat_arr)
end

"""
    isapprox(m1, m2)

Return whether two matrices are approximately identical.
"""
function isapprox(m1::u1_matrix, m2::u1_matrix)
    blk_chk(m1, m2) && isapprox(m1.submat_arr, m2.submat_arr)
end

"""
    +(m1, m2)

Add two matrices.
"""
function +(m1::u1_matrix, m2::u1_matrix)
    if ! blk_chk(m1, m2)
        throw(error("block structure mismatch"))
    end
    u1_matrix(m1.d_vec, m1.qn, m1.submat_arr + m2.submat_arr)
end

"""
    -(m1, m2)

Calculate m1 - m2.
"""
function -(m1::u1_matrix, m2::u1_matrix)
    if ! blk_chk(m1, m2)
        throw(error("block structure mismatch"))
    end
    u1_matrix(m1.d_vec, m1.qn, m1.submat_arr - m2.submat_arr)
end

"""
    *(m, x)

Number-Matrix multiplication.
"""
function *(m::u1_matrix, x::Real)
    submat_arr = [subm .* x for subm in m.submat_arr]
    u1_matrix(m.d_vec, m.qn, submat_arr)
end
"""
    *(x, m)
Number-Matrix multiplication.
"""
function *(x::Real, m::u1_matrix)
    m*x
end

"""
    tr(m)

Trace of matrix.
"""
function tr(m::u1_matrix)
    if m.qn != 0
        return 0.0
    end
    trace_value = 0.0
    for subm in m.submat_arr
        trace_value += tr(subm)
    end
    trace_value
end

"""
    ⊗(m1, m2)

Tensor product of matrices.
"""
function ⊗(m1::u1_matrix, m2::u1_matrix)
    kron(m1, m2)
end

"""
    kron(m1, m2)

Tensor product of matrices.
"""
function kron(m1::u1_matrix, m2::u1_matrix)
    d_vec, bookkeeping_func = kron_bookkeeping(m1.d_vec, m2.d_vec)

    qn = m1.qn + m2.qn

    m1_l_labels = l_labels_f(length(m1.d_vec), m1.qn)
    m2_l_labels = l_labels_f(length(m2.d_vec), m2.qn)
    llen1, llen2 = length(m1_l_labels), length(m2_l_labels)

    l1_train = reshape( m1_l_labels' .* ones(Int, llen2), llen1*llen2)
    l2_train = reshape( ones(Int, llen1)' .* m2_l_labels, llen1*llen2)

    m = zero_u1m(Float64, d_vec, qn)
    for (l1, l2) in zip(l1_train, l2_train)
        r1, r2 = l1 + m1.qn, l2 + m2.qn
        subm1, subm2 = submat(m1, l1), submat(m2, l2)
        l, l_index0, l_index1 = bookkeeping_func(l1, l2)
        r, r_index0, r_index1 = bookkeeping_func(r1, r2)
        subm_to_add = kron(subm1, subm2)
        add_to_submat!(m, subm_to_add, l, l_index0:l_index1, r_index0:r_index1)
    end
    m
end

"""
    *(m1, m2)

Matrix multiplication.
"""
function *(m1::u1_matrix{Ti, Tf}, m2::u1_matrix{Ti, Tf}) where {Ti, Tf}
    if m1.d_vec != m2.d_vec
        throw(error("block structures are not identitcal"))
    end
    d_vec = m1.d_vec
    qn = m1.qn + m2.qn
    l_labels = l_labels_f(length(d_vec), qn)
    l1_labels = l_labels_f(length(d_vec), m1.qn)
    l2_labels = l_labels_f(length(d_vec), m2.qn)

    m = zero_u1m(Tf, d_vec, qn)
    for l in l_labels
        if (l in l1_labels) && (l+m1.qn in l2_labels)
            subm1 = submat(m1, l)
            subm2 = submat(m2, l + m1.qn)
            add_to_submat!(m, subm1 * subm2, l)
        end
    end
    m
end

@doc raw"""
    reflect(m)

reflect a u1_matrix. Equivalent to the following unitary tranformation 
```math
\left( \begin{array}{cccc}
  &  &  & 1\\
  &  & 1 & \\
  & \udots &  & \\
  1 &  &  & 
\end{array} \right)
```
"""
function reflect(m::u1_matrix)
    #if m.d_vec != reverse(m.d_vec)
    #    throw(error("this matrix is not reflectable!"))
    #end

    submat_arr = [subm[end:-1:1, end:-1:1] for subm in m.submat_arr]
    submat_arr = submat_arr[end:-1:1]
    u1_matrix(reverse(m.d_vec), -m.qn, submat_arr)
end

"""
    transpose(m)

Take the transpose of a u1_matrix.
"""
function transpose(m::u1_matrix{Ti, Tf}) where {Ti, Tf}
    submat_arr = [Array{Tf}(subm') for subm in m.submat_arr]
    u1_matrix(m.d_vec, -m.qn, submat_arr)
end

"""
    adjoint(m)

This function does nothing to the matrix. 
It is useful when calculating the adjoint (transpose) of a cMPO.  
"""
function adjoint(m::u1_matrix)
    m
end

"""
    symmetrize(m)
    
Symmetrize the matrix by 
```math
(M + M^T) / 2
```
"""
function symmetrize(m::u1_matrix)
    if m.qn != 0
        throw(error("input matrix is not block-diagonalized"))
    end
    submat_arr = [0.5*(subm + transpose(subm)) for subm in m.submat_arr]
    u1_matrix(m.d_vec, m.qn, submat_arr)
end

"""
    eigh(m)

Eigendecomposition of a (symmetric) U1 matrix.
"""
function eigh(m::u1_matrix{Ti, Tf}) where {Ti, Tf}

    if m.qn != 0 #|| m != transpose(m)
        throw(error("input matrix is not symmetric, block-diagonalized matrix"))
    end

    w = Array{Tf, 1}([])
    v = zero_u1m(Tf, m.d_vec, 0)
    for (ix, subm) in enumerate(m.submat_arr)
        w1, v1 = eigen(Symmetric(subm))
        w = vcat(w, w1)
        add_to_submat!(v, v1, ix)
    end
    w, v
end

@doc raw"""
    log_tr_expm(m, beta)

Calculates 
```math
\log \mathrm{Tr} \exp(\beta M)
```
"""
function log_tr_expm(m::u1_matrix, beta::AbstractFloat)
    m = symmetrize(m)
    w, _ = eigh(m)
    logsumexp(w .* beta)
end

"""
    add_to_submat!(m, subm_to_add, l_label, l_indices, r_indices)

Inplace modification of the U1 matrix `m` by adding some small matrix `subm_to_add` to it.
The small matrix is added to the block corresponding to the quantum number `l_label`, 
within the sub-block indicated by the index ranges `l_indices` and `r_indices`.
"""
function add_to_submat!(m::u1_matrix, subm_to_add::Array{<:AbstractFloat, 2}, l_label::Integer, l_indices::UnitRange{<:Integer}, r_indices::UnitRange{<:Integer})
    qabs, qsgn = abs(m.qn), sign(m.qn)
    submat_index = l_label - (qabs * (1 - qsgn) ÷ 2)
    m.submat_arr[submat_index][l_indices, r_indices] += subm_to_add
    m
end
"""
    add_to_submat!(m, subm_to_add, l_label)

Inplace modification of the U1 matrix `m` by adding some small matrix `subm_to_add` to it.
The small matrix is added to the whole block corresponding to the quantum number `l_label`. 
"""
function add_to_submat!(m::u1_matrix, subm_to_add::Array{<:AbstractFloat, 2}, l_label::Integer)
    qabs, qsgn = abs(m.qn), sign(m.qn)
    submat_index = l_label - (qabs * (1 - qsgn) ÷ 2)
    m.submat_arr[submat_index] += subm_to_add
    m
end

"""
    to_array(m)

Represent the U1 matrix as an ordinary 2D-array.
"""
function toarray(m::u1_matrix{Ti, Tf}) where {Ti, Tf}
    dim = sum(m.d_vec)
    nlen = length(m.d_vec)

    index0_vec = ones(Ti, nlen)
    index1_vec = ones(Ti, nlen)
    d_accum = 0
    for (ix, d) in enumerate(m.d_vec)
        index0_vec[ix] += d_accum
        d_accum += d
        index1_vec[ix] += d_accum -1
    end

    arr = zeros(Tf, dim, dim)
    l_labels = l_labels_f(nlen, m.qn)
    r_labels = l_labels .+ m.qn

    for (ix, (l, r)) in enumerate(zip(l_labels, r_labels))
        l_index0, l_index1 = index0_vec[l], index1_vec[l]
        r_index0, r_index1 = index0_vec[r], index1_vec[r]
        arr[l_index0:l_index1, r_index0:r_index1] += m.submat_arr[ix]
    end
    arr

end

"""
    vec(u1_matrix, is_diag=false)

Convert all the parameters (the data within the blocks) to a vector. 
If `is_diag` is set to `true`, then only take the diagonal elements.
"""
function vec(m::u1_matrix{Ti, Tf}, is_diag::Bool = false) where {Ti, Tf}
   params = Array{Tf, 1}([])
   subm_to_vec = if is_diag diag else vec end

   for subm in m.submat_arr
       params = vcat(params, subm_to_vec(subm))
   end
   params
end

"""
    params_len(d_vec, qn)

Given the quantum number sector `d_vec` and the quantum number of the matrix `qn`,
determine the total number of the parameters.
"""
function params_len(d_vec::Array{<:Integer, 1}, qn::Integer)
    qabs = abs(qn)
    sum(d_vec[1:end-qabs] .* d_vec[1+qabs:end])
end

"""
    l_labels_f(nlen, qn)

Given the total number `nlen` of the U(1) quantum numbers and the quantum number `qn` of the U1 matrix, 
return a vector that maps the index of the block to the U(1) quantum number of the block. 
For example, when qn=1, the first block corresponds to quantum number 1.
when qn=-1, however, the first block corresponds to quantum number 2. 
(The U(1) quantum number is assumed to start from 1 to `nlen`.) 
"""
function l_labels_f(nlen::Integer, qn::Integer)
    qabs, qsgn = abs(qn), sign(qn)
    (1:nlen-qabs) .+ (qabs * (1 - qsgn) ÷ 2)
end

"""
    submat(m, n)

Return the block corresponding to U(1) quantum number n. 
"""
function submat(m::u1_matrix, n::Integer)
    qabs, qsgn = abs(m.qn), sign(m.qn)
    submat_index = n - (qabs * (1 - qsgn) ÷ 2)
    m.submat_arr[submat_index]
end

"""
    kron_bookkeeping(d_vec1, d_vec2)

Utilization function for the calculation of the tensor product. 
Given two U(1) quantum number sectors, return 
- the new quantum number sector for the tensor product matrix. 
- a function that maps the qunatum numbers `(n1, n2)` corresponding to input quantum number sectors 
  to the sub-block location `(l, x0, x1)` within the new matrix, where `l` is the new quantum number
  and `x0` and `x1` represent the beginning and ending point within in the block.
"""
function kron_bookkeeping(d_vec1::Array{<:Integer, 1}, d_vec2::Array{<:Integer, 1})
    nlen1, nlen2 = length(d_vec1), length(d_vec2)
    nlen = nlen1 + nlen2 - 1

    n1_train = reshape((1:nlen1)' .* ones(Int, nlen2), nlen1*nlen2)
    n2_train = reshape(ones(Int, nlen1)' .* (1:nlen2), nlen1*nlen2)
    ntot_train = n1_train + n2_train .- 1

    index0_train = ones(Int, nlen1*nlen2)
    index1_train = ones(Int, nlen1*nlen2)
    d_vec =  zeros(Int, nlen)

    for ix in 1:nlen1*nlen2
        ntot = ntot_train[ix]
        index0_train[ix] += d_vec[ntot]
        d_vec[ntot] += d_vec1[n1_train[ix]] * d_vec2[n2_train[ix]]
        index1_train[ix] += d_vec[ntot] - 1
    end

    function bookkeeping_func(n1::Integer, n2::Integer)
        ix = (n1-1) * nlen2 + (n2-1) + 1
        ntot_train[ix], index0_train[ix], index1_train[ix]
    end

    d_vec, bookkeeping_func
end

"""
    logsumexp(w)

Calculation of logsumexp.
"""
function logsumexp(w::Array{<:AbstractFloat, 1})
    u = maximum(w)
    t = 0.0
    for i = 1:length(w)
        t += exp(w[i]-u)
    end
    u + log(t)
end

"""
    isometry_truncate(m, v, msk_vec)

Truncate the U1 matrix `m` by an isometry matrix. 
We implement this by first transform the matrix by a orthogonal matrix `v` 
that has the same block structure with `m`, and then truncate the transformed U1 matrix 
according to the mask `msk_vec`, block by block. 
"""
function isometry_truncate(m::u1_matrix{Ti, Tf}, v::u1_matrix{Ti, Tf}, msk_vec::Vector{BitVector}) where {Ti, Tf}
    trunc_d_vec = map(sum, msk_vec)

    m1 = transpose(v) * m * v
    newm = zero_u1m(Tf, trunc_d_vec[trunc_d_vec .> 0], m1.qn)

    l_labels = l_labels_f(length(trunc_d_vec), m1.qn)
    new_l_labels = l_labels_f(length(newm.d_vec), m1.qn)
    newix = 0
    for (ix, subm) in enumerate(m1.submat_arr)
        l_label = l_labels[ix]
        r_label = l_label + m1.qn
        if (m1.qn >= 0 && trunc_d_vec[l_label] > 0) || (m1.qn < 0 && trunc_d_vec[r_label] > 0)
            newix += 1
        end
        if trunc_d_vec[l_label] != 0 && trunc_d_vec[r_label] != 0
            new_l_label = new_l_labels[newix]
            add_to_submat!(newm, subm[msk_vec[l_label], msk_vec[r_label]], new_l_label)
        end
    end
    newm
end

"""
    trunc_msk_gen(d_vec, w, num_kept)

Generate `msk_vec` for the function `isometry_truncate(m, v, msk_vec)`.
According to the block structure `d_vec` and some spectrum `w`, keep `num_kept` of states in the spectrum.
Return a vector of BitVector that tells how to truncate a U1 matrix, block by block.
"""
function trunc_msk_gen(d_vec::Vector{<:Integer}, w::Vector{<:AbstractFloat}, num_kept::Integer)
    w_sorted = sort(w)
    threshold = 0.5 * (w_sorted[end-num_kept] + w_sorted[end-num_kept+1])

    index = 1
    trunc_msk_vec = Vector{BitVector}([])

    for d in d_vec
        subw = w[index:index+d-1]
        index = index + d
        push!(trunc_msk_vec, subw .> threshold)
    end
    trunc_msk_vec
end
