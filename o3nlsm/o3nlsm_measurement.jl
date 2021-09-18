"""
calculate energy and the effective Hamiltonian spectrum 
"""

using U1cMPO
using JLD
using Printf
using Optim
using ArgParse

current_dir = dirname(@__FILE__)
include(current_dir*"/o3nlsm_utils.jl")

# read the command line arguments
s = ArgParseSettings()
@add_arg_table! s begin
    "--dataname"
        help = "jld file that stores the rawdata"
        arg_type = String
        default = "None"
end
parsed_args = parse_args(s)
dataname = parsed_args["dataname"]

#beta = 20.0
#K = 1.1
#double_lmax = 3
#chi = 12
#dataname = @sprintf("rawdata_o3pi_K=%.1f_beta=%.2f_lmax=%.1f_chi=%d.jld", K, beta, double_lmax/2, chi)

# load previous parameters
prev_result = load(dataname) # load jld from previous dataname
beta = prev_result["beta"]
K = prev_result["K"]
double_lmax = prev_result["double_lmax"]
d_vec = prev_result["d_vec"]
chi = sum(d_vec)

# cmpo 
T = cmpo_gen(K, double_lmax)

# load previous optimized cMPS params
params = prev_result["optimized_params"]  
# params to cMPS converter
_, params_to_cmps, _ = cmps_funcs_gen(d_vec)

# calculate energy
psi = params_to_cmps(params)
energy_value = energy(T, psi, beta)

# calculate the spectrum of K_{|-|} and K_{|-|-|}
# named as `gong` and `wang` 
# since `|-|` `|-|-|` look like Chinese characters gong and wang
K_gong = reflect(psi) * psi
K_wang = reflect(psi) * (T * psi)
w_gong, _ = eigh(K_gong)
w_wang, v_wang = eigh(K_wang)

# calculate the correlation function in the imaginary direction
Id = eye_u1m(Float64, d_vec)
O_wang = reflect(Id) ⊗ X_theta_pi(0, double_lmax) ⊗ Id
O_wang = transpose(v_wang) * O_wang * v_wang
w0 = logsumexp(w_wang .* beta) / beta
corrs = Vector{Float64}([])
for tau in (1:99) .* (beta / 100)
    expw1 = init_diag_u1m(K_wang.d_vec, exp.((w_wang .- w0) .* tau))
    expw2 = init_diag_u1m(K_wang.d_vec, exp.((w_wang .- w0) .* (beta-tau)))
    corr = U1cMPO.tr(expw1 * O_wang * expw2 * O_wang)
    push!(corrs, corr)
end

# sort the spectrum 
w_gong = sort(w_gong)
w_wang = sort(w_wang)

# save
measfile_name = @sprintf("measurements_o3pi_g=%.1f_beta=%.2f_lmax=%.1f_chi=%d.jld", g, beta, double_lmax/2, chi)
save(measfile_name,
    "beta", beta,
    "K", K,
    "double_lmax", double_lmax,
    "d_vec", d_vec,
    "energy_value", energy_value,
    "spectrum_|-|", w_gong,
    "spectrum_|-|-|", w_wang,
    "corrs", corrs,
)