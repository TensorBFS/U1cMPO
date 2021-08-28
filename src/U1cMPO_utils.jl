using Zygote

"""
    free_energy(T, psi, beta)

given cMPS `psi`, *Hermitian* cMPO `T` and the inversed temperature `beta`, 
calculate the free energy. 
"""
function free_energy(T::cmpo, psi::cmps, beta::AbstractFloat)
    Tpsi = T * psi
    f = -( ovlp(reflect(psi), Tpsi, beta) - ovlp(reflect(psi), psi, beta) ) / beta
    f
end

"""
    f_and_gf_gen(T, params_to_cmps, beta)

Given a function `params_to_cmps` that convert parameter vector to cmps,
the *Hermtian* cMPO `T`, and the inversed temperature `beta`, 
generate functions that calculate free energy and the corresponding gradient from parameters.
"""
function f_and_gf_gen(T::cmpo, params_to_cmps::Function, beta::AbstractFloat)
    function f(params::Vector{<:AbstractFloat})
        psi = params_to_cmps(params)
        free_energy(T, psi, beta)
    end
    function gf!(gparams::Vector{<:AbstractFloat}, params::Vector{<:AbstractFloat})
        gparams[1:end] = gradient(params -> f(params), params)[1][1:end]
    end
    f, gf!
end
"""
    fidel_and_gfidel_gen(phi, params_to_cmps, beta)

Given a function `params_to_cmps` that convert parameter vector to cmps,
the target cMPS `phi`, and the inversed temperature `beta`, 
generate functions that calculate free energy and the corresponding gradient from parameters.
"""
function fidel_and_gfidel_gen(phi::cmps, params_to_cmps::Function, beta::AbstractFloat)
    function fidel(params::Vector{<:AbstractFloat})
        psi = params_to_cmps(params)
        - ovlp(reflect(psi), phi, beta) + 0.5 * (ovlp(reflect(psi), psi, beta))
    end
    function gfidel!(gparams::Vector{<:AbstractFloat}, params::Vector{<:AbstractFloat})
        gparams[1:end] = gradient(params -> fidel(params), params)[1][1:end]
    end
    fidel, gfidel!
end

"""
    energy(T, psi, beta)

Calculate energy from cMPO `T`, cMPS `psi` at inversed temperature `beta`.
"""
function energy(T::cmpo, psi::cmps, beta::AbstractFloat)
    Tpsi = T * psi

    rho1 = reflect(psi) * Tpsi
    rho2 = reflect(psi) * psi

    w1, _ = eigh(rho1)
    wnm1 = w1 .- logsumexp(w1 .* beta) / beta

    w2, _ = eigh(rho2)
    wnm2 = w2 .- logsumexp(w2 .* beta) / beta

    energy_value = - sum(exp.(wnm1 .* beta) .* w1) + sum(exp.(wnm2 .* beta) .* w2)
    energy_value
end

"""
    klein(Lpsi, psi, beta)

Calculate the klein bottle entropy from left cMPS `Lpsi` and right cMPS `psi` at 
inversed temperature `beta`. 
"""
function klein(Lpsi::cmps, psi::cmps, beta::AbstractFloat)
    d_vec = psi.Q.d_vec
    rho_d_vec, bookkeeping_func = U1cMPO.kron_bookkeeping(d_vec, d_vec)

    perm_list = Vector{Vector{Int64}}([])
    for _ in 1:length(rho_d_vec)
        push!(perm_list, Vector{Int64}([]))
    end

    for ix in 1:length(d_vec)
        for iy in 1:length(d_vec)
            subm_index, a, b = bookkeeping_func(iy, ix)
            perm = vec(reshape(a:b, (d_vec[ix], d_vec[iy]))')
            perm_list[subm_index] = vcat(perm_list[subm_index], perm)
        end
    end

    rho = Lpsi * psi
    w, v = eigh(rho)
    w = w .- logsumexp(w .* beta) / beta
    exp_rho = v * init_diag_u1m(rho_d_vec, exp.(w .* beta/2)) * transpose(v)

    trace_value = 0.0
    for (ix, subm) in enumerate(exp_rho.submat_arr)
        trace_value += tr(subm[perm_list[ix], 1:end])
    end
    2*log(trace_value)
end

"""
    C_matrix(psi, beta, tau)

A util function for the the calculation of the von-Neumann entropy which calculates the coefficent matrix. 
`psi` is cMPS, `beta` is the inversed temperature, and `tau` is the interval in the imaginary time.
We have inserted some unitary matrices to maintain the U(1) block structure of the matrix. 
"""
function C_matrix(psi::cmps{Ti, Tf}, beta::AbstractFloat, tau::AbstractFloat) where {Ti, Tf}
    rho = reflect(psi) * psi
    w, v = eigh(rho)
    w = w .- logsumexp(w .* beta) / beta

    d_vec = psi.Q.d_vec
    rvsd_d_vec = d_vec[end:-1:1]
    rho_d_vec, bookkeeping_func = U1cMPO.kron_bookkeeping(rvsd_d_vec, d_vec)
    exp_rho = v * init_diag_u1m(rho_d_vec, exp.(w .* tau)) * transpose(v)

    # reverse bookkeeping
    sectors = []
    for _ in 1:length(rho_d_vec)
        push!(sectors, [])
    end
    for l1 in 1:length(rvsd_d_vec)
        for l2 in 1:length(d_vec)
            l, a, b = bookkeeping_func(l1, l2)
            push!(sectors[l], (a, b, l1, l2))
        end
    end

    # construct C-matrix
    M = zero_u1m(Tf, rho_d_vec, 0)
    for (l, subm) in enumerate(exp_rho.submat_arr)
        for (al, bl, l1, l2) in sectors[l]
            for (ar, br, r1, r2) in sectors[l]
                newl2, newr1 = length(d_vec)+1-l2, length(d_vec)+1-r1

                sub_subm = reshape(subm[al:bl, ar:br], (d_vec[l2], rvsd_d_vec[l1], d_vec[r2], rvsd_d_vec[r1]))
                #sub_subm = sub_subm[1:end, end:-1:1, end:-1:1, 1:end]
                sub_subm = sub_subm[end:-1:1, 1:end, 1:end, end:-1:1]
                sub_subm = ein"badc->cadb"(sub_subm)
                sub_subm = reshape(sub_subm, (rvsd_d_vec[r1]*rvsd_d_vec[l1], d_vec[r2]*d_vec[l2]))

                newl, new_al, new_bl = bookkeeping_func(l1, newr1)
                newr, new_ar, new_br = bookkeeping_func(newl2, r2)
                add_to_submat!(M, sub_subm, newl, new_al:new_bl, new_ar:new_br)
            end
        end
    end
    M
end
"""
    von_neumann_entropy(psi, beta, tau)

For cMPS `psi`, calculate the von-Neumann entropy between intervals τ and β-τ.
"""
function von_neumann_entropy(psi::cmps, beta::AbstractFloat, tau::AbstractFloat)
    M1 = C_matrix(psi, beta, tau)
    M2 = C_matrix(psi, beta, beta-tau)
    d_vec = M1.d_vec

    w1, v1 = eigh(M1)
    w1[w1 .< 0] .= 0.0
    r_rho = init_diag_u1m(d_vec, sqrt.(w1)) *
            transpose(v1) * M2 * v1 *
            init_diag_u1m(d_vec, sqrt.(w1))

    sw, _ = eigh(r_rho)
    sw = sw[sw .> 1e-12]
    -sum(sw .* log.(sw))
end

"""
    reduced_density_matrix(psi, beta)

For cMPS `psi` with imaginary-time length `beta`, construct the reduced density matrix.
We have inserted some unitaries to maintain the U(1) block structure of the matrix.
"""
function reduced_density_matrix(psi::cmps{Ti, Tf}, beta::AbstractFloat) where {Ti, Tf}
    rho = reflect(psi) * psi
    w, v = eigh(rho)
    w = w .- logsumexp(w .* beta) / beta

    d_vec = psi.Q.d_vec
    rvsd_d_vec = d_vec[end:-1:1]
    rho_d_vec, bookkeeping_func = U1cMPO.kron_bookkeeping(rvsd_d_vec, d_vec)
    exp_rho = v * init_diag_u1m(rho_d_vec, exp.(w .* beta)) * transpose(v)

    # reverse bookkeeping
    sectors = []
    for _ in 1:length(rho_d_vec)
        push!(sectors, [])
    end
    for l1 in 1:length(rvsd_d_vec)
        for l2 in 1:length(d_vec)
            l, a, b = bookkeeping_func(l1, l2)
            push!(sectors[l], (a, b, l1, l2))
        end
    end

    # partial trace
    reduced_rho = zero_u1m(Tf, d_vec, 0)
    for (l, subm) in enumerate(exp_rho.submat_arr)
        for (al, bl, l1, l2) in sectors[l]
            for (ar, br, r1, r2) in sectors[l]
                if l1 == r1
                    sub_subm = reshape(subm[al:bl, ar:br], (d_vec[l2], rvsd_d_vec[l1], d_vec[r2], rvsd_d_vec[r1]))
                    sub_subm = ein"bada->bd"(sub_subm)
                    add_to_submat!(reduced_rho, sub_subm, l2)
                end
            end
        end
    end
    reduced_rho
end

"""
    truncate_according_to_rdm(psi, chi, beta)

Truncate cMPS `psi` to target bond dimension `chi` at inversed temperature `beta` 
according to the reduced density matrix eigenvalues. 
The new U(1) quantum number sectors can be automatically determined from this process.
"""
function truncate_according_to_rdm(psi::cmps, chi::Integer, beta::AbstractFloat)
    d_vec = psi.Q.d_vec
    reduced_rho = reduced_density_matrix(psi, beta)
    w, v = eigh(reduced_rho)
    msk_vec = trunc_msk_gen(d_vec, w, chi)
    Q = isometry_truncate(psi.Q, v, msk_vec)
    R = map(Rx -> isometry_truncate(Rx, v, msk_vec), psi.R)

    diagQ(cmps(Q, R))
end
