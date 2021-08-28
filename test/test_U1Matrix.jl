using LinearAlgebra

@testset "basic arithmetic" begin
    d_vec1 = [1,3,3,1]
    d_vec2 = [1,2,2,1]
    params1a = sin.(1:15)
    params1b = cos.(1:15)
    params2 = Array{Float64}(1:8)

    m1a = init_u1m_by_params(d_vec1, 1, params1a)
    m1b = init_u1m_by_params(d_vec1, 1, params1b)
    m2 = init_u1m_by_params(d_vec2, 1, params2)

    m1apb = init_u1m_by_params(d_vec1, 1, params1a+params1b)
    m1amb = init_u1m_by_params(d_vec1, 1, params1a-params1b)
    m1at3 = init_u1m_by_params(d_vec1, 1, params1a*3)

    @test m1a == m1a
    @test m1a != m1b
    @test m1a != m2
    @test U1cMPO.blk_chk(m1a, m1b)
    @test U1cMPO.blk_chk(m1a+m1b, m1apb)
    @test (m1a+m1b) == m1apb
    @test (m1a-m1b) == m1amb
    @test m1a*3 == m1at3
end

@testset "basic utilities" begin
    d_vec = [1,2,2,1]
    params0 = Array{Float64}(1:10)
    params1 = Array{Float64}(1:8)
    params2 = Array{Float64}(1:4)

    @test params_len(d_vec, 0) == 10
    @test params_len(d_vec, 1) == 8
    @test params_len(d_vec, 2) == 4
    @test params_len(d_vec, -2) == 4

    @test U1cMPO.l_labels_f(4, 0) == Array{Int}(1:4)
    @test U1cMPO.l_labels_f(4, 1) == Array{Int}(1:3)
    @test U1cMPO.l_labels_f(4, -2) == Array{Int}(3:4)

    arr0 = Array{Float64}([1  0  0  0  0  0
                           0  2  4  0  0  0
                           0  3  5  0  0  0
                           0  0  0  6  8  0
                           0  0  0  7  9  0
                           0  0  0  0  0  10])
    arr1 = Array{Float64}([0  1  2  0  0  0
                           0  0  0  3  5  0
                           0  0  0  4  6  0
                           0  0  0  0  0  7
                           0  0  0  0  0  8
                           0  0  0  0  0  0])
    arr2 = Array{Float64}([0  0  0  0  0  0
                           0  0  0  0  0  0
                           0  0  0  0  0  0
                           1  0  0  0  0  0
                           2  0  0  0  0  0
                           0  3  4  0  0  0])
    U1cMPO.l_labels_f(4, -2) .+ (-2)
    m0 = init_u1m_by_params(d_vec, 0, params0)
    m1 = init_u1m_by_params(d_vec, 1, params1)
    m2 = init_u1m_by_params(d_vec, -2, params2)

    @test toarray(m0) == arr0
    @test toarray(m1) == arr1
    @test toarray(m2) == arr2
end

@testset "bookkeeping" begin
    d_vec = [1,3,3,1]
    d_vec_tnp_0 = [1,6,15,20,15,6,1]
    d_vec_tnp, bookkeeping_func = U1cMPO.kron_bookkeeping(d_vec, d_vec)
    @test d_vec_tnp_0 == d_vec_tnp

    @test bookkeeping_func(1, 1) == (1, 1, 1)
    @test bookkeeping_func(1, 2) == (2, 1, 3)
    @test bookkeeping_func(2, 1) == (2, 4, 6)
    @test bookkeeping_func(1, 3) == (3, 1, 3)
    @test bookkeeping_func(2, 2) == (3, 4, 12)
    @test bookkeeping_func(3, 1) == (3, 13, 15)
    @test bookkeeping_func(1, 4) == (4, 1, 1)
    @test bookkeeping_func(2, 3) == (4, 2, 10)
    @test bookkeeping_func(3, 2) == (4, 11, 19)
    @test bookkeeping_func(4, 1) == (4, 20, 20)
    @test bookkeeping_func(2, 4) == (5, 1, 3)
    @test bookkeeping_func(3, 3) == (5, 4, 12)
    @test bookkeeping_func(4, 2) == (5, 13, 15)
    @test bookkeeping_func(3, 4) == (6, 1, 3)
    @test bookkeeping_func(4, 3) == (6, 4, 6)
    @test bookkeeping_func(4, 4) == (7, 1, 1)
end

@testset "kron and eigh" begin
    d_vec = [1,3,3,1]
    params = sin.(1:15)

    mp = init_u1m_by_params(d_vec, 1, params)
    mm = init_u1m_by_params(d_vec, -1, params)
    mp1 = transpose(mm)
    mm1 = transpose(mp)

    mtot = kron(mp, mm) + kron(mm1, mp1)
    w, _ = eigh(mtot)
    w = sort(w)

    arr_p = toarray(mp)
    arr_m = toarray(mm)

    arr_tot = kron(arr_p, arr_m) + kron(arr_p', arr_m')
    w1 = eigvals(arr_tot)

    @test isapprox(w, w1; atol=10^-12)
    @test kron(mp, mm) == mp ⊗ mm
end

@testset "reflect" begin
    d_vec = [1,3,2,1]
    params = sin.(1:11)
    m = init_u1m_by_params(d_vec, 1, params)

    X = zeros(Float64, 7, 7)
    for ix in 1:7
        X[ix, 8-ix] += 1
    end

    @test all(X * toarray(m) * X .== toarray(reflect(m)))
end

@testset "matrix multiplication" begin
    d_vec = [1, 3, 3, 1]
    m1 = init_u1m_by_params(d_vec, 1, rand(15))
    m2 = init_u1m_by_params(d_vec, -1, rand(15))
    m3 = init_u1m_by_params(d_vec, 0, rand(20))

    @test isapprox(toarray(m1*m2), toarray(m1)*toarray(m2))
    @test isapprox(toarray(m1*m3), toarray(m1)*toarray(m3))
end

@testset "truncation by isometries" begin
    num_kept = 4

    d_vec = [1,3,3,1]
    plen = params_len(d_vec, 0)
    m = init_u1m_by_params(d_vec, 0, rand(plen))
    m = symmetrize(m)
    w, v = eigh(m)
    trunc_msk_vec = trunc_msk_gen(d_vec, w, num_kept)

    w_arr, v_arr = eigen(Symmetric(toarray(m)))

    m_trunc = isometry_truncate(m, v, trunc_msk_vec)
    w_trunc, _ = eigh(m_trunc)
    @test isapprox(w_arr[end-num_kept+1:end], sort(w_trunc))

    plen1 = params_len(d_vec, 1)
    m1 = init_u1m_by_params(d_vec, 1, rand(plen1))
    m1_trunc = isometry_truncate(m1, v, trunc_msk_vec)
    vec1 = vec(m1_trunc)
    vec1 = vec1[sortperm(abs.(vec1))]

    w1, _ = eigh(m1_trunc * transpose(m1_trunc))
    w1 = sort(w1)

    m1_trunc_arr = transpose(v_arr)[end-num_kept+1:end, 1:end] * toarray(m1) * v_arr[1:end, end-num_kept+1:end]
    vec1_arr = vec(m1_trunc_arr)
    vec1_arr = vec1_arr[sortperm(abs.(vec1_arr))][end-length(vec1)+1:end]

    w1_arr = eigvals(Symmetric(m1_trunc_arr * transpose(m1_trunc_arr)))

    @test isapprox(vec1, vec1_arr)
    @test isapprox(w1, w1_arr)
end
