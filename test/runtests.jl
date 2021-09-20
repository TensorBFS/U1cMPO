using Revise
using U1cMPO
using Test

@testset "U1Matrix.jl" begin
    include("test_U1Matrix.jl")
end

@testset "U1MatrixChainRules.jl" begin
    include("test_U1MatrixChainRules.jl")
end

@testset "sto_julia_wrapper.jl" begin
    include("test_sto_julia_wrapper.jl")
end

@testset "pauli_matrices.jl" begin
    include("test_pauli_matrices.jl")
end

@testset "U1cMPO_basic.jl" begin
    include("test_U1cMPO_basic.jl")
end

@testset "U1cMPO_utils.jl" begin
    include("test_U1cMPO_utils.jl")
end
