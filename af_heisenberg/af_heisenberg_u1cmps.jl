using U1cMPO

using Optim
using JLD
using Printf
using ArgParse

current_dir = dirname(@__FILE__)
include(current_dir*"/af_heisenberg_utils.jl")

# read the command line arguments
s = ArgParseSettings()
@add_arg_table! s begin
    "--beta"
        help = "beta"
        arg_type = Float64
        default = 32.0
    "--chi"
        help = "chi"
        arg_type = Int64
        default = 4
    "--init"
        help = "init guess for cMPS, if None, use random init"
        arg_type = String
        default = "None"
end
parsed_args = parse_args(s)

beta = parsed_args["beta"]
chi = parsed_args["chi"]
init = parsed_args["init"]
# set the parameters manually for test
#beta = 8.0
#chi = 6
#init = "rawdata_af_heisenberg_beta=8.00_chi=4.jld"

# generate the target data name
dataname = @sprintf("rawdata_af_heisenberg_beta=%.2f_chi=%d.jld", beta, chi)

# cmpo 
T = cmpo_gen()

# initialization
psi = cmps(T.Q, T.L)
if init != "None"
    global psi
    prev_result = load(init)
    prev_params = prev_result["optimized_params"]
    prev_d_vec = prev_result["d_vec"]
    _, prev_params_to_cmps, _ = cmps_funcs_gen(prev_d_vec)
    psi = prev_params_to_cmps(prev_params)
end

# optimization
f_list = Vector{AbstractFloat}([])
d_vec_list = [] # d_vec: U(1) quantum number sectors
psi_list = [] # psi: the compressed cMPS at each power step

lowest_f = Inf
num_failed_power_step = 0
# power method
for ix in 1:25
    global psi, lowest_f, num_failed_power_step
    optimized_f, d_vec, psi = power_step(T, psi, chi, beta)
    push!(f_list, optimized_f)
    push!(d_vec_list, d_vec)
    push!(psi_list, psi)
    println("step ", ix)
    println("optimized_f ", optimized_f)
    println("d_vec ", d_vec)

    if optimized_f < lowest_f
        lowest_f = optimized_f
    else
        num_failed_power_step += 1
    end

    if num_failed_power_step >= 2
        break
    end
end
best_loc = argmin(f_list)
optimized_f = f_list[best_loc]
d_vec = d_vec_list[best_loc]
optimized_psi = psi_list[best_loc]

# variational optimization
println("--------------------------------------------------------")
println("lowest free energy found at power step ", best_loc)
println("best d_vec ", d_vec)
println("current lowest free energy ", optimized_f)
println("ready to perform variational optimization")
println("--------------------------------------------------------")
optimized_f, optimized_psi = variational_optim(T, optimized_psi, beta)

_, _, cmps_to_params = cmps_funcs_gen(d_vec)
optimized_params = cmps_to_params(optimized_psi)

optimized_Lpsi = psi_to_Lpsi(optimized_psi)
klein_value = klein(optimized_Lpsi, optimized_psi, beta)
von_neumann_entropy_value = von_neumann_entropy(optimized_psi, beta, beta/2)
energy_value = energy(T, optimized_psi, beta)

println("--------------------------------------------------------")
println("d_vec", d_vec)
println("optimized_f ", optimized_f)
println("klein_value ", klein_value)
println("von_neumann_entropy_value ", von_neumann_entropy_value)
println("energy_value ", energy_value)
println("--------------------------------------------------------")

# data saving
save(dataname,
    "optimized_params", optimized_params,
    "optimized_f", optimized_f,
    "d_vec", d_vec,
    "beta", beta,
    "klein_value", klein_value,
    "von_neumann_entropy_value", von_neumann_entropy_value,
    "energy_value", energy_value
)
