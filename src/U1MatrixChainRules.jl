using ChainRulesCore
using ChainRules
using LinearAlgebra
using OMEinsum

import ChainRulesCore: rrule, frule

"""
rules missed in LinearAlgebra
"""
function rrule(::typeof(kron), x::Matrix{Tf}, y::Matrix{Tf}) where Tf <: Real
    z = kron(x, y)

    xdim1, xdim2 = size(x)
    ydim1, ydim2 = size(y)

    function kron_pushback(z̄)
        z̄_tmp = reshape(z̄, ydim1, xdim1, ydim2, xdim2)
        x̄ = ein"abcd,ac->bd"(z̄_tmp, y)
        ȳ = ein"abcd,bd->ac"(z̄_tmp, x)
        NoTangent(), x̄, ȳ
    end

    z, kron_pushback
end

"""
rules for U1Matrix
"""
function rrule(::typeof(init_u1m_by_params), d_vec::Vector{<:Integer}, qn::Integer, params::Vector{<:AbstractFloat})
    fwd = init_u1m_by_params(d_vec, qn, params)

    function param_init_pushback(f̄wd)
        NoTangent(), NoTangent(), NoTangent(), vec(f̄wd)
    end

    fwd, param_init_pushback
end

function rrule(::typeof(init_diag_u1m), d_vec::Array{Ti, 1}, params::Array{Tf, 1}) where {Ti<:Integer, Tf<:AbstractFloat}
    fwd = init_diag_u1m(d_vec, params)

    function diag_init_pushback(f̄wd)
        p̄arams = Array{Tf, 1}([])
        for s̄ubm in f̄wd.submat_arr
            p̄arams = vcat(p̄arams, diag(s̄ubm))
        end
        NoTangent(), NoTangent(), p̄arams
    end

    fwd, diag_init_pushback
end

@non_differentiable eye_u1m(::Type{<:AbstractFloat}, ::Array{<:Integer, 1})
@non_differentiable zero_u1m(::Type{<:AbstractFloat}, ::Array{<:Integer, 1}, ::Integer)
@non_differentiable blk_chk(::u1_matrix, ::u1_matrix)
@non_differentiable ==(::u1_matrix, ::u1_matrix)

function rrule(::typeof(+), m1::u1_matrix, m2::u1_matrix)
    fwd = m1 + m2
    function plus_pushback(f̄wd)
        NoTangent(), f̄wd, f̄wd
    end

    fwd, plus_pushback
end

function rrule(::typeof(-), m1::u1_matrix, m2::u1_matrix)
    fwd = m1 - m2
    function minus_pushback(f̄wd)
        NoTangent(), f̄wd, f̄wd * -1.0
    end

    fwd, minus_pushback
end

function rrule(::typeof(*), m::u1_matrix, x::Real)
    fwd = m * x
    function times_pushback(f̄wd)
        m̄ = f̄wd * x
        x̄ = sum(vec(f̄wd) .* vec(m))
        NoTangent(), m̄, x̄
    end

    fwd, times_pushback
end

function rrule(::typeof(*), x::Real, m::u1_matrix)
    fwd = x * m
    function times_pushback(f̄wd)
        m̄ = f̄wd * x
        x̄ = sum(vec(f̄wd) .* vec(m))
        NoTangent(), x̄, m̄
    end
    fwd, times_pushback
end

function rrule(::typeof(⊗), m1::u1_matrix{Ti, Tf}, m2::u1_matrix{Ti, Tf}) where {Ti<:Integer, Tf<:AbstractFloat}
    d_vec, bookkeeping_func = kron_bookkeeping(m1.d_vec, m2.d_vec)

    qn = m1.qn + m2.qn
    nlen = length(d_vec)

    m1_l_labels = l_labels_f(length(m1.d_vec), m1.qn)
    m2_l_labels = l_labels_f(length(m2.d_vec), m2.qn)
    llen1, llen2 = length(m1_l_labels), length(m2_l_labels)

    l1_train = reshape( m1_l_labels' .* ones(Int, llen2), llen1*llen2)
    l2_train = reshape( ones(Int, llen1)' .* m2_l_labels, llen1*llen2)
    subm_kron_pushback_train = []

    fwd = zero_u1m(Float64, d_vec, qn)
    for (l1, l2) in zip(l1_train, l2_train)
        r1, r2 = l1 + m1.qn, l2 + m2.qn
        subm1, subm2 = submat(m1, l1), submat(m2, l2)
        l, l_index0, l_index1 = bookkeeping_func(l1, l2)
        r, r_index0, r_index1 = bookkeeping_func(r1, r2)
        subm_to_add, subm_kron_pushback = rrule(kron, subm1, subm2)
        add_to_submat!(fwd, subm_to_add, l, l_index0:l_index1, r_index0:r_index1)
        push!(subm_kron_pushback_train, subm_kron_pushback)
    end

    function kron_pushback(f̄wd) 
        _, bookkeeping_func = kron_bookkeeping(m1.d_vec, m2.d_vec)
        m̄1 = zero_u1m(Tf, m1.d_vec, m1.qn)
        m̄2 = zero_u1m(Tf, m2.d_vec, m2.qn)

        for (l1, l2, subm_kron_pushback) in zip(l1_train, l2_train, subm_kron_pushback_train)
            r1, r2 = l1 + m1.qn, l2 + m2.qn
            l, l_index0, l_index1 = bookkeeping_func(l1, l2)
            r, r_index0, r_index1 = bookkeeping_func(r1, r2)
            s̄ubm = submat(f̄wd, l)[l_index0:l_index1, r_index0:r_index1]
            _, s̄ubm1, s̄ubm2 = subm_kron_pushback(s̄ubm)

            add_to_submat!(m̄1, s̄ubm1, l1)
            add_to_submat!(m̄2, s̄ubm2, l2)
        end

        NoTangent(), m̄1, m̄2
    end

    fwd, kron_pushback
end

rrule(::typeof(kron), m1::u1_matrix, m2::u1_matrix) = rrule(⊗, m1, m2)

function rrule(::typeof(*), m1::u1_matrix{Ti, Tf}, m2::u1_matrix{Ti, Tf}) where {Ti, Tf}
    if m1.d_vec != m2.d_vec
        throw(error("block structures are not identitcal"))
    end
    d_vec = m1.d_vec
    qn = m1.qn + m2.qn
    l_labels = l_labels_f(length(d_vec), qn)
    l1_labels = l_labels_f(length(d_vec), m1.qn)
    l2_labels = l_labels_f(length(d_vec), m2.qn)

    subm_multiply_pushbacks = []
    valid_l_labels = []
    fwd = zero_u1m(Tf, d_vec, qn)
    for l in l_labels
        if (l in l1_labels) && (l+m1.qn in l2_labels)
            subm1 = submat(m1, l)
            subm2 = submat(m2, l + m1.qn)
            subm, subm_multiply_pushback = rrule(*, subm1, subm2)
            push!(subm_multiply_pushbacks, subm_multiply_pushback)
            push!(valid_l_labels, l)
            add_to_submat!(fwd, subm1 * subm2, l)
        end
    end

    function multiply_pushback(f̄wd)
        m̄1 = zero_u1m(Tf, m1.d_vec, m1.qn)
        m̄2 = zero_u1m(Tf, m2.d_vec, m2.qn)
        for (l, subm_multiply_pushback) in zip(valid_l_labels, subm_multiply_pushbacks)
            s̄ubm = submat(f̄wd, l)
            _, s̄ubm1, s̄ubm2 = subm_multiply_pushback(s̄ubm)
            add_to_submat!(m̄1, unthunk(s̄ubm1), l)
            add_to_submat!(m̄2, unthunk(s̄ubm2), l+m1.qn)
        end
        NoTangent(), m̄1, m̄2
    end
    fwd, multiply_pushback
end

function rrule(::typeof(reflect), m::u1_matrix)
    fwd = reflect(m)

    function reflect_pushback(f̄wd)
        NoTangent(), reflect(f̄wd)
    end

    fwd, reflect_pushback
end

function rrule(::typeof(transpose), m::u1_matrix)
    fwd = transpose(m)

    function transpose_pushback(f̄wd)
        NoTangent(), transpose(f̄wd)
    end

    fwd, transpose_pushback
end

function rrule(::typeof(adjoint), m::u1_matrix)
    fwd = m
    function adjoint_pushback(f̄wd)
        NoTangent(), f̄wd
    end
    fwd, adjoint_pushback
end

function rrule(::typeof(symmetrize), m::u1_matrix)
    fwd = symmetrize(m)
    function symmetrize_pushback(f̄wd)
        NoTangent(), symmetrize(f̄wd)
    end
    fwd, symmetrize_pushback

end

function rrule(::typeof(log_tr_expm), m::u1_matrix{Ti, Tf}, beta::AbstractFloat) where {Ti, Tf}
    symm, symm_pushback = rrule(symmetrize, m)

    w, v = eigh(symm)
    fwd = logsumexp(w .* beta)

    expw = init_diag_u1m(m.d_vec, exp.((beta .* w) .- fwd ))
    scaled_rho = v * expw * transpose(v)

    function log_tr_expm_pushback(f̄wd)
        w̄ = transpose(scaled_rho) * beta * f̄wd
        _, w̄ = symm_pushback(w̄)
        b̄eta = tr(scaled_rho * symm) * f̄wd
        NoTangent(), w̄, b̄eta
    end

    fwd, log_tr_expm_pushback
end

@non_differentiable params_len(::Array{<:Integer, 1}, ::Integer)
@non_differentiable l_labels_f(::Integer, ::Integer)
@non_differentiable kron_bookkeeping(::Array{<:Integer, 1}, ::Array{<:Integer, 1})
@non_differentiable trunc_msk_gen(::Vector{<:Integer}, ::Vector{<:AbstractFloat}, ::Integer)

# we avoid implementing these functions,
# although they are actually differentiable
@non_differentiable eigh(::u1_matrix)
@non_differentiable tr(::u1_matrix)
@non_differentiable add_to_submat!(::u1_matrix, ::Array{<:AbstractFloat, 2}, ::Integer, ::UnitRange{<:Integer}, ::UnitRange{<:Integer})
@non_differentiable add_to_submat!(::u1_matrix, ::Array{<:AbstractFloat, 2}, ::Integer)
@non_differentiable toarray(::u1_matrix)
@non_differentiable vec(::u1_matrix)
@non_differentiable submat(::u1_matrix, n::Integer)
@non_differentiable isometry_truncate(::u1_matrix, ::u1_matrix, ::Vector{BitVector})
