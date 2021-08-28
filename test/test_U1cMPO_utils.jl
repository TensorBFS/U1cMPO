using FiniteDifferences
using LinearAlgebra
using OMEinsum

# generate cMPO for XY model, for test
sp = pauli(1)
sm = pauli(-1)
sz = pauli(0)
Q = zero_u1m(Float64, sz.d_vec, 0)
L = [sp, sm]
R = [sm, sp]
T = cmpo(Q, L, R)

# cmps info
d_vec = [1,2,3,4]
#d_vec = [1,1,1,1]
plen_Q = sum(d_vec)
plen_R = params_len(d_vec, 1)
chi = plen_Q

function params_to_cmps(params::Vector{<:AbstractFloat})
    Q = init_diag_u1m(d_vec, params[1:plen_Q])
    R1 = init_u1m_by_params(d_vec, 1, params[plen_Q+1:end])
    R2 = transpose(R1)
    psi = cmps(Q, [R1, R2])
end

function check_f_and_gf(params::Vector{<:AbstractFloat}, beta::AbstractFloat)
    psi = params_to_cmps(params)
    f, gf! = f_and_gf_gen(T, params_to_cmps, 1.0)

    check1 = free_energy(T, psi, 1.0) == f(params)
    gparams = zeros(Float64, length(params))
    gf!(gparams, params)
    check2 = isapprox(grad(central_fdm(5, 1), f, params)[1] , gparams)
    check1 && check2
end
@test check_f_and_gf(rand(plen_Q + plen_R), 4.2)
@test check_f_and_gf(rand(plen_Q + plen_R), 1.0)
@test check_f_and_gf(rand(plen_Q + plen_R), 16.0)

# energy
function check_energy(params::Vector{<:AbstractFloat}, beta::AbstractFloat)
    psi = params_to_cmps(params)
    
    # calculate energy density by numerical diff with respect to logZ
    dbeta = 1e-8
    energyNumDiff = (free_energy(T, psi, beta+dbeta) * (beta + dbeta) - free_energy(T, psi, beta) * beta) / dbeta

    # calculate energy density by calling function energy 
    energyDirect = energy(T, psi, beta)
    println("direct ", energyDirect) 
    println("numdiff ", energyNumDiff)
    
    # compare
    isapprox(energyNumDiff, energyDirect, rtol=sqrt(dbeta))
end 
@test check_energy(rand(plen_Q + plen_R), 1.2)
@test check_energy(rand(plen_Q + plen_R), 3.4)
@test check_energy(rand(plen_Q + plen_R), 5.6)

# klein bottle entropy
function check_klein_bottle_entropy(params::Vector{<:AbstractFloat}, beta::AbstractFloat)
    psi = params_to_cmps(params)
    Id = eye_u1m(psi)
    Q = psi.Q
    R1, R2 = psi.R
    Lpsi = cmps(Q, [R2, R1])

    rho_arr = kron(toarray(Q), toarray(Id)) +
              kron(toarray(Id), toarray(Q)) +
              kron(toarray(R1), toarray(R2)) +
              kron(toarray(R2), toarray(R1))
    w, v = eigen(Symmetric(rho_arr))
    #isapprox(v * Diagonal(w) * v', rho_arr)
    w = w .- (logsumexp(w .* beta) ./ beta)
    exp_rho_arr = v * Diagonal(exp.(w .* beta/2)) * transpose(v)
    exp_rho_arr = reshape(exp_rho_arr, (chi, chi, chi, chi))

    klein_arr = 2*log(ein"abba->"(exp_rho_arr)[1])
    klein_value = klein(Lpsi, psi, beta)
    isapprox(klein_arr, klein_value)
end

@test check_klein_bottle_entropy(rand(plen_Q + plen_R), 1.2)
@test check_klein_bottle_entropy(rand(plen_Q + plen_R), 3.4)
@test check_klein_bottle_entropy(rand(plen_Q + plen_R), 5.6)

# test von-Neumann entropy
function check_von_Neumann_entropy(params::Vector{<:AbstractFloat}, beta::AbstractFloat, tau::AbstractFloat)
    psi = params_to_cmps(params)
    Id = eye_u1m(psi)
    Q = psi.Q
    R1, R2 = psi.R
    rho_arr = kron(toarray(Q), toarray(Id)) +
              kron(toarray(Id), toarray(Q)) +
              kron(toarray(R1), toarray(R1)) +
              kron(toarray(R2), toarray(R2))
    w, v = eigen(Symmetric(rho_arr))
    w = w .- (logsumexp(w .* beta) ./ beta)
    M1 = v * Diagonal(exp.(w .* tau)) * transpose(v)
    M1 = reshape(M1, (chi, chi, chi, chi))
    M1 = reshape(ein"badc->cadb"(M1), (chi^2, chi^2))
    @assert isapprox(M1, M1')
    w1, v1 = eigen(Symmetric(M1))

    M2 = v * Diagonal(exp.(w .* (beta-tau))) * transpose(v)
    M2 = reshape(M2, (chi, chi, chi, chi))
    M2 = reshape(ein"badc->cadb"(M2), (chi^2, chi^2))

    r_rho = Diagonal(sqrt.(w1)) * transpose(v1) * M2 * v1 * Diagonal(sqrt.(w1))
    sw = eigvals(Symmetric(r_rho))
    sw = sw[sw .> 1e-12]
    entropy_from_arr = -sum(sw .* log.(sw))
    entropy_value = von_neumann_entropy(psi, beta, tau)

    #entropy_value, entropy_from_arr
    isapprox(entropy_from_arr, entropy_value)
end

@test check_von_Neumann_entropy(rand(plen_Q + plen_R), 8.0, 4.2)
@test check_von_Neumann_entropy(rand(plen_Q + plen_R), 8.0, 1.3)
@test check_von_Neumann_entropy(rand(plen_Q + plen_R), 8.0, 5.7)

# test reduced density matrix
function check_reduced_density_matrix(params::Vector{<:AbstractFloat}, beta::AbstractFloat)
    psi = params_to_cmps(params)
    Id = eye_u1m(psi)
    Q = psi.Q
    R1, R2 = psi.R
    rho_arr = kron(toarray(Q), toarray(Id)) +
              kron(toarray(Id), toarray(Q)) +
              kron(toarray(R1), toarray(R1)) +
              kron(toarray(R2), toarray(R2))
    w, v = eigen(Symmetric(rho_arr))
    w = w .- (logsumexp(w .* beta) ./ beta)
    rho_arr = v * Diagonal(exp.(w .* beta)) * transpose(v)
    rho_arr = reshape(rho_arr, (chi, chi, chi, chi))
    reduced_rho_arr = ein"bada->bd"(rho_arr)
    w_rho_from_arr = eigvals(Symmetric(reduced_rho_arr))

    reduced_rho = reduced_density_matrix(psi, beta)
    w_rho, _ = eigh(reduced_rho)
    w_rho = sort(w_rho)
    isapprox(w_rho, w_rho_from_arr)
end

@test check_reduced_density_matrix(rand(plen_Q + plen_R), 4.2)
@test check_reduced_density_matrix(rand(plen_Q + plen_R), 1.0)
@test check_reduced_density_matrix(rand(plen_Q + plen_R), 8.0)
@test check_reduced_density_matrix(rand(plen_Q + plen_R), 16.0)

# truncate_according_to_rdm not tested
