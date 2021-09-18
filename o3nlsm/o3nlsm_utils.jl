using U1cMPO
using Printf

"""
    cmpo_gen(K, double_lmax)

Generate cMPO for quantum rotor model described by O(3) nonlinear simga model with θ=π. 
`K` means the coupling between neighboring rotors.
`double_lmax` means the twice of the cut of angular momentum.
""" 
function cmpo_gen(K::AbstractFloat, double_lmax::Integer)
    Xm = X_theta_pi(-1, double_lmax)
    Xp = X_theta_pi(1, double_lmax)
    X0 = X_theta_pi(0, double_lmax)
    L2 = L2_theta_pi(double_lmax)

    Q_T = L2 * (-1.0 / 2.0 / K)
    L_T = [-sqrt(K) * Xm, -sqrt(K) * Xp, -sqrt(K) * X0]
    R_T = [ sqrt(K) * Xp,  sqrt(K) * Xm,  sqrt(K) * X0]
    cmpo(Q_T, L_T, R_T)
end

"""
    cmps_funcs_gen(d_vec)

Given U(1) quantum number sector, generate utilization functions for cMPS calculations.
Return one integer and two functions: 
- `plen_tot`: the total number of paramters in the cMPS.
- `params_to_cmps`: function that convert paramter array to cMPS
- `cmps_to_params`: function that convert cMPS to paramter array
"""
function cmps_funcs_gen(d_vec::Vector{<:Integer})
    plen_Q = sum(d_vec)
    plen_Rp = params_len(d_vec, 1)
    plen_Rz = params_len(d_vec, 0)
    plen_tot = plen_Q + plen_Rp + plen_Rz
    function params_to_cmps(params::Array{<:AbstractFloat, 1})
        Q = init_diag_u1m(d_vec, params[1:plen_Q])
        Rp = init_u1m_by_params(d_vec, 1, params[plen_Q+1:plen_Q+plen_Rp])
        Rz = init_u1m_by_params(d_vec, 0, params[plen_Q+plen_Rp+1:end])
        Rz = symmetrize(Rz)
        #if d_vec == d_vec[end:-1:1]
        #    Q = 0.5 * (Q + reflect(Q))
        #    Rp = 0.5 * (Rp - 1.0 * transpose(reflect(Rp)))
        #    Rz = 0.5 * (Rz - reflect(Rz))
        #end
        Rm = -1.0 * transpose(Rp)
        cmps(Q, [Rm, Rp, Rz])
    end
    function cmps_to_params(psi::cmps)
        vcat(vec(psi.Q, true), vec(psi.R[2]), vec(psi.R[3]))
    end
    plen_tot, params_to_cmps, cmps_to_params
end

"""
    psi_to_Lpsi(psi)

Given a right boundary cMPS, return the corresponding left boundary cMPS according to the structure of the cMPO.
"""
function psi_to_Lpsi(psi::cmps)
    cmps(psi.Q, [psi.R[2] * -1.0, psi.R[1] * -1.0, psi.R[3] * -1.0])
end

"""
    power_step(T, psi, chi, beta)

Given target bond dimension `chi` and inversed temperature `beta`, perform one power method step: act the cMPO `T` onto the input cMPS `psi`, compress it, and return the new cMPS.
"""
function power_step(T::cmpo, psi::cmps, chi::Integer, beta::AbstractFloat)
    Tpsi = T * psi
    psi1 = diagQ(Tpsi)

    if sum(Tpsi.Q.d_vec) > chi
        psi1 = truncate_according_to_rdm(Tpsi, chi, beta)
    end

    d_vec1 = psi1.Q.d_vec

    _, params_to_cmps1, cmps_to_params1 = cmps_funcs_gen(d_vec1)
    params1 = cmps_to_params1(psi1)

    if sum(Tpsi.Q.d_vec) > chi
        fidel, gfidel! = fidel_and_gfidel_gen(Tpsi, params_to_cmps1, beta)
        res_fidel = optimize(fidel, gfidel!, params1, LBFGS(), Optim.Options(iterations=200))
        params1 = Optim.minimizer(res_fidel)
    end

    psi1 = params_to_cmps1(params1)

    free_energy(T, psi1, beta), d_vec1, psi1
end

@doc raw"""
    variational_optim(T, psi, beta)

Given the *Hermitian* cMPO `T` and the inversed temperature `beta`, variationally optimize 
```math
    \frac{\langle \psi |T|\psi \rangle}{\langle \psi | \psi \rangle}.
```
The variational optimization is initialized with the input `psi`.
"""
function variational_optim(T::cmpo, psi::cmps, beta::AbstractFloat)
    d_vec = psi.Q.d_vec
    _, params_to_cmps, cmps_to_params = cmps_funcs_gen(d_vec)
    params = cmps_to_params(psi)

    f, gf! = f_and_gf_gen(T, params_to_cmps, beta)
    res_f = optimize(f, gf!, params, LBFGS(), Optim.Options(show_trace=true, iterations=1000))

    psi1 = params_to_cmps(Optim.minimizer(res_f))

    Optim.minimum(res_f), psi1
end
