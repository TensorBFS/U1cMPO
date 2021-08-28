using ChainRulesCore
using ChainRules
using ChainRulesTestUtils

#ammendments to make ChainRulesTestUtils work for u1_matrix
import ChainRulesTestUtils: rand_tangent, check_equal
using Random
function rand_tangent(rng::AbstractRNG, x::u1_matrix)
    Δsubmat_arr = rand_tangent(x.submat_arr)
    u1_matrix(x.d_vec, x.qn, Δsubmat_arr)
end
function check_equal(x::u1_matrix, y::u1_matrix; kwargs...)
    isapprox(x, y)
end

using FiniteDifferences
import FiniteDifferences: to_vec
function to_vec(x::u1_matrix)
    v, Array_from_vec = to_vec(x.submat_arr)
    function u1_matrix_from_vec(x_vec)
        subms = Array_from_vec(x_vec)
        u1_matrix(x.d_vec, x.qn, subms)
    end
    v, u1_matrix_from_vec
end

#test kron rule
@testset "kron" begin
    test_rrule(kron, rand(2,3), rand(3,2))
    test_rrule(kron, rand(4,5), rand(3,2))
end

#test u1_matrix ChainRules
@testset "init_by_params" begin
    d_vec = [1, 3, 3, 1]
    qn = 1
    params = rand(15)
    test_rrule(init_u1m_by_params, d_vec ⊢ nothing, qn ⊢ nothing, params)
end

@testset "diag" begin
    d_vec = [1, 3, 3, 1]
    params = rand(8)
    test_rrule(init_diag_u1m, d_vec ⊢ nothing, params)
end

@testset "plus" begin
    d_vec, qn = [1, 3, 3, 1], 1
    params1 = rand(15)
    params2 = rand(15)
    m1 = init_u1m_by_params(d_vec, qn, params1)
    m2 = init_u1m_by_params(d_vec, qn, params2)
    test_rrule(+, m1, m2)
end

@testset "minus" begin
    d_vec, qn = [1, 3, 3, 1], 1
    params1 = rand(15)
    params2 = rand(15)
    m1 = init_u1m_by_params(d_vec, qn, params1)
    m2 = init_u1m_by_params(d_vec, qn, params2)
    test_rrule(-, m1, m2)
end

@testset "multiply float" begin
    d_vec, qn = [1, 3, 3, 1], 1
    params = rand(15)
    m = init_u1m_by_params(d_vec, qn, params)
    test_rrule(*, m, 4.2)
    test_rrule(*, 4.2, m)
end

@testset "kron" begin
    d_vec = [1, 3, 3, 1]
    m1 = init_u1m_by_params(d_vec, 1, rand(15))
    m2 = init_u1m_by_params(d_vec, 1, rand(15))
    test_rrule(⊗, m1, m2)
    d_vec = [1, 3, 3, 1]
    m1 = init_u1m_by_params(d_vec, 1, rand(15))
    m2 = init_u1m_by_params(d_vec, -1, rand(15))
    test_rrule(⊗, m1, m2)
    d_vec = [1, 3, 3, 1]
    m1 = init_u1m_by_params(d_vec, 2, rand(6))
    m2 = init_u1m_by_params(d_vec, -1, rand(15))
    test_rrule(⊗, m1, m2)
    d_vec = [1, 3, 3, 1]
    m1 = init_u1m_by_params(d_vec, 0, rand(20))
    m2 = init_u1m_by_params(d_vec, -1, rand(15))
    test_rrule(kron, m1, m2)
end

@testset "multiply" begin
    d_vec = [1, 3, 3, 1]
    m1 = init_u1m_by_params(d_vec, 1, rand(15))
    m2 = init_u1m_by_params(d_vec, 1, rand(15))
    test_rrule(*, m1, m2)
    d_vec = [1, 3, 3, 1]
    m1 = init_u1m_by_params(d_vec, 1, rand(15))
    m2 = init_u1m_by_params(d_vec, -1, rand(15))
    test_rrule(*, m1, m2)
    d_vec = [1, 3, 3, 1]
    m1 = init_u1m_by_params(d_vec, 2, rand(6))
    m2 = init_u1m_by_params(d_vec, -1, rand(15))
    test_rrule(*, m1, m2)
    d_vec = [1, 3, 3, 1]
    m1 = init_u1m_by_params(d_vec, 0, rand(20))
    m2 = init_u1m_by_params(d_vec, -1, rand(15))
    test_rrule(*, m1, m2)
end

@testset "reflect" begin
    d_vec = [1, 3, 3, 1]
    m = init_u1m_by_params(d_vec, 0, rand(20))
    test_rrule(reflect, m)
    d_vec = [1, 2, 3, 1]
    m = init_u1m_by_params(d_vec, 1, rand(11))
    test_rrule(reflect, m)
end

@testset "transpose" begin
    d_vec = [1, 3, 3, 1]
    m = init_u1m_by_params(d_vec, 0, rand(20))
    test_rrule(transpose, m)
    d_vec = [1, 2, 3, 1]
    m = init_u1m_by_params(d_vec, 1, rand(11))
    test_rrule(transpose, m)
end

@testset "adjoint" begin
    d_vec = [1, 3, 3, 1]
    m = init_u1m_by_params(d_vec, 0, rand(20))
    test_rrule(adjoint, m)
end

@testset "symmetrize" begin
    d_vec = [1, 3, 3, 1]
    m = init_u1m_by_params(d_vec, 0, rand(20))
    test_rrule(symmetrize, m)
end

@testset "log_tr_expm" begin
    d_vec = [1, 3, 3, 1]
    m = init_u1m_by_params(d_vec, 0, rand(20))
    test_rrule(log_tr_expm, m, 4.2)

    d_vec = [1, 2, 3, 2, 1]
    m = init_u1m_by_params(d_vec, 0, rand(19))
    test_rrule(log_tr_expm, m, 4.2)
end
