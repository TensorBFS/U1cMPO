using PyCall
src_directory = joinpath(dirname(@__FILE__), "..", "src")
pushfirst!(PyVector(pyimport("sys")."path"), src_directory)

@testset "sto for theta=0" begin
    sto = pyimport("spherical_tensor_operator")

    permorder = [4,1,3,2]
    X = sto.X_theta_0(1, 1)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(X_theta_0(1, 1)) == X
    X = sto.X_theta_0(0, 1)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(X_theta_0(0, 1)) == X
    X = sto.X_theta_0(-1, 1)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(X_theta_0(-1, 1)) == X
    X = sto.L2_theta_0(1)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(L2_theta_0(1)) == X

    permorder = [9, 4, 8, 1, 3, 7, 2, 6, 5]
    X = sto.X_theta_0(1, 2)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(X_theta_0(1, 2)) == X
    X = sto.X_theta_0(0, 2)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(X_theta_0(0, 2)) == X
    X = sto.X_theta_0(-1, 2)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(X_theta_0(-1, 2)) == X
    X = sto.L2_theta_0(2)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(L2_theta_0(2)) == X

end

@testset "sto for theta=pi" begin
    sto = pyimport("spherical_tensor_operator")

    permorder = [2,1]
    X = sto.X_theta_pi(1, 0.5)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(X_theta_pi(1, 1)) == X
    X = sto.X_theta_pi(0, 0.5)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(X_theta_pi(0, 1)) == X
    X = sto.X_theta_pi(-1, 0.5)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(X_theta_pi(-1, 1)) == X
    X = sto.L2_theta_pi(0.5)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(L2_theta_pi(1)) == X

    permorder = [6, 2, 5, 1, 4, 3]
    X = sto.X_theta_pi(1, 1.5)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(X_theta_pi(1, 3)) == X
    X = sto.X_theta_pi(0, 1.5)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(X_theta_pi(0, 3)) == X
    X = sto.X_theta_pi(-1, 1.5)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(X_theta_pi(-1, 3)) == X
    X = sto.L2_theta_pi(1.5)
    X = X[permorder, 1:end][1:end, permorder]
    @test toarray(L2_theta_pi(3)) == X

end
