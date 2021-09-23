# U1cMPO

[![CI](https://github.com/TensorBFS/U1cMPO/actions/workflows/ci.yml/badge.svg)](https://github.com/TensorBFS/U1cMPO/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/TensorBFS/U1cMPO/branch/main/graph/badge.svg?token=TTbadaxozU)](https://codecov.io/gh/TensorBFS/U1cMPO)

Performing a tensor network simulation of the nonlinear sigma model (NLSM) with $\theta=\pi$ topological term with Julia. 

This repository includes

- a package that implements $U(1)$ symmetry in the [continuous matrix product operator (cMPO)](https://arxiv.org/abs/2004.12928) method
- Simulation and measurements for the NLSM with $\theta=\pi$. 

## Installation

- Download the repository by 

	```bash
	git clone https://github.com/TensorBFS/U1cMPO.git
	```
	
- Enter the repository, and open Julia's interactive session (known as REPL). Press `]` to enter the pkg mode, and then type the following command 

	```julia
	pkg> activate .
	pkg> instantiate
	```
	
	then exit the REPL. 

## Usage

To simulate the NLSM with $\theta=\pi$ term with the cMPO method, we map it to a modified quantum rotor model 
$$
a \hat{H} = \sum_j \frac{(\hat{\boldsymbol{L}}'_j)^2}{2 K} + K \sum_{\langle i, j \rangle} \hat{\boldsymbol{n}}_i \cdot \hat{\boldsymbol{n}}_j \,.
$$
where $\hat{\boldsymbol{L}}'_j$ is the angular momentum operator decorated by a magnetic monopole, $\hat{\boldsymbol{n}}$ is the rotor operator, $a$ is the lattice spacing, and $K>0$ is a constant. In the low-energy and long-wavelength limit, the field theoretical description of this rotor model is the $O(3)$ NLSM with $\theta=\pi$ and $1/g^2=K$. The tensor network representation of this modified quantum rotor model is derived using the monopole harmonics basis. More details can be found in [this manuscript](https://arxiv.org/abs/21xx.xxxxx).

To run the simulation, enter the repository and type the following command

```bash
julia --project=. o3nlsm/o3nlsm_u1cmps.jl --beta 8 --K 2 --doublelmax 3 --chi 8
```

which will perform the simulation for $K=2.0$ and inverse temperature $\beta=8$. Moreover, the monopole harmonics basis will be truncated at $l_{\mathrm{max}}=3/2$, and the bond dimension of the boundary cMPS is $\chi=8$. The results (including the optimized cMPS, the free energy, and the bipartite entanglement entropy of the boundary cMPS) will be saved in a hdf5-format file named as `rawdata_o3pi_K=2.0_beta=8.00_lmax=1.5_chi=8.jld`. With this file, we can either perform further measurements like

```bash
julia --project=. o3nlsm/o3nlsm_measurement.jl --dataname rawdata_o3pi_K=2.0_beta=8.00_lmax=1.5_chi=8.jld
```

or use this datafile to provide the initial guess for the simulations with larger bond dimensions, such as

```bash
julia --project=. o3nlsm/o3nlsm_u1cmps.jl --beta 8 --K 2 --doublelmax 3 --chi 12 --init rawdata_o3pi_K=2.0_beta=8.00_lmax=1.5_chi=8.jld
```

Alternatively, the U1cMPO package can also be used to calculate the finite-temperature properties of other models with $U(1)$ symmetry.

