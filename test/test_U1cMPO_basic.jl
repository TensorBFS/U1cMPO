using LinearAlgebra

d_vec = [1,3,3,1]
paramsQ = sin.(1:20)
Q = init_u1m_by_params(d_vec, 0, paramsQ)
Q = symmetrize(Q)

paramsR = sin.(1:15)
R1 = init_u1m_by_params(d_vec, 1, paramsR)
R2 = transpose(R1)
R = [R1, R2]
I = eye_u1m(Float64, d_vec)

psi = cmps(Q, R)

d_vec_pauli = [1,1]
params_p = [1.0]
params_z = [1.0, -1.0]
Sp = init_u1m_by_params(d_vec_pauli, 1, params_p)
Sm = transpose(Sp)
Sz = init_u1m_by_params(d_vec_pauli, 0, params_z)

Q0 = init_u1m_by_params(d_vec_pauli, 0, [0.0, 0.0])
L0 = [Sp, Sm]
R0 = [Sm, Sp]
I0 = eye_u1m(Float64, d_vec_pauli)
#P0 = [Q0 Q0; Q0 Q0]

op = cmpo(Q0, L0, R0)

# test cmpo * cmps
tpsi = op * psi
@test tpsi.Q == I0 ⊗ Q + Q0 ⊗ I + Sp ⊗ R2 + Sm ⊗ R1
@test tpsi.R[1] == Sp ⊗ I
@test tpsi.R[2] == Sm ⊗ I

# test cmps * cmps
rho = reflect(psi) * psi
@test rho == transpose(rho)

w, _ = eigh(rho)
w = sort(w)

Q_arr = toarray(Q)
R1_arr = toarray(R1)
R2_arr = toarray(R2)
I_arr = toarray(I)
rho_arr = kron(Q_arr, I_arr) + kron(I_arr, Q_arr) + kron(R1_arr, R1_arr) + kron(R2_arr, R2_arr)

w1 = eigvals(rho_arr)

@test isapprox(w, w1; atol=10^-12)

# test cmps * cmpo
psit = reflect(psi) * op
@test psit.Q == reflect(Q) ⊗ I0 + I ⊗ Q0 + reflect(R2) ⊗ Sm + reflect(R1) ⊗ Sp
@test psit.R[1] == I ⊗ Sm
@test psit.R[2] == I ⊗ Sp

# test cmpo * cmpo
tt = op * op
rho1 = reflect(psi) * tt * psi
rho2 = reflect(psi) * op * tpsi
rho3 = reflect(tpsi) * tpsi

w1, _ = eigh(rho1)
w2, _ = eigh(rho2)
w3, _ = eigh(rho3)
@test all(isapprox.(w1, w2, atol=10^-12))
@test all(isapprox.(w1, w3, atol=10^-12))

@test isapprox(log_tr_expm(rho1, 1.23), log_tr_expm(rho2, 1.23))
@test isapprox(log_tr_expm(rho1, 1.23), log_tr_expm(rho3, 1.23))
