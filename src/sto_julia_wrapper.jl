#Utilization functions for the matrices in O(3) quantum rotor models.
using PyCall

"""
    d_vec_0(lmax)

Return the U(1) block structure given the largest l kept in the case of no theta term.
"""
function d_vec_0(lmax::Integer)
    [1:(lmax+1); lmax:-1:1]
end

"""
    X_theta_0(M, lmax)

Return matrix representation of spherical tensor operator for ordinary quantum rotors.
"""
function X_theta_0(M::Integer, lmax::Integer)
    d_vec = d_vec_0(lmax)
    qn = M

    src_directory = dirname(@__FILE__)
    pushfirst!(PyVector(pyimport("sys")."path"), src_directory)
    sto = pyimport("spherical_tensor_operator")

    # m1 = M + m2
    submat_arr = Array{Array{Float64, 2}, 1}([])
    for m1 in lmax:-1:(-lmax)
        m2 = m1 - qn
        if -lmax <= m2 <= lmax
            calc_elem(l1::AbstractFloat, l2::AbstractFloat) = sto.Xelem_theta_0(qn, l1, m1, l2, m2)
            l1s = abs(m1):lmax
            l2s = abs(m2):lmax
            l1_blk = ones(Float64, length(l2s))' .* l1s
            l2_blk = l2s' .* ones(Float64, length(l1s))
            push!(submat_arr, map(calc_elem, l1_blk, l2_blk))
        end
    end

    u1_matrix(d_vec, qn, submat_arr)
end

"""
    L2_theta_0(lmax)

Return the matrix representation for the kinetic term in quantum rotor model (no theta term).
"""
function L2_theta_0(lmax::Integer)
    d_vec = d_vec_0(lmax)
    params = Array{Float64, 1}([])
    for m in lmax:-1:-lmax
        params = vcat(params, map(l->l*(l+1), abs(m):lmax))
    end
    init_diag_u1m(d_vec, params)
end

"""
    d_vec_pi(lmax)

Return the U(1) block structure given the largest l kept in the case of theta=pi.
"""
function d_vec_pi(double_lmax::Integer)
    if double_lmax % 2 == 0
        throw(error("lmax should be half integer"))
    end
    [1:(double_lmax+1)÷2 ; (double_lmax+1)÷2:-1:1]
end

"""
    X_theta_pi(M, lmax)

Return matrix representation of spherical tensor operator for quantum rotors 
decorated with a magnetic monopole (theta=pi).
"""
function X_theta_pi(M::Integer, double_lmax::Integer)
    d_vec = d_vec_pi(double_lmax)
    qn = M

    src_directory = dirname(@__FILE__)
    pushfirst!(PyVector(pyimport("sys")."path"), src_directory)
    sto = pyimport("spherical_tensor_operator")

    lmax = double_lmax / 2
    # m1 = M + m2
    submat_arr = Array{Array{Float64, 2}, 1}([])
    for m1 in lmax:-1:(-lmax)
        m2 = m1 - qn
        if -lmax <= m2 <= lmax
            calc_elem(l1::AbstractFloat, l2::AbstractFloat) = sto.Xelem_theta_pi(qn, l1, m1, l2, m2)
            l1s = abs(m1):lmax
            l2s = abs(m2):lmax
            l1_blk = ones(Float64, length(l2s))' .* l1s
            l2_blk = l2s' .* ones(Float64, length(l1s))
            push!(submat_arr, map(calc_elem, l1_blk, l2_blk))
        end
    end

    u1_matrix(d_vec, qn, submat_arr)
end

"""
    L2_theta_0(lmax)

Return the matrix representation for the kinetic term in quantum rotor model (theta=pi).
"""
function L2_theta_pi(double_lmax::Integer)
    d_vec = d_vec_pi(double_lmax)
    lmax = double_lmax / 2
    params = Array{Float64, 1}([])
    for m in lmax:-1:-lmax
        params = vcat(params, map(l->l*(l+1), abs(m):lmax))
    end
    init_diag_u1m(d_vec, params)
end
